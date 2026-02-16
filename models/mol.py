from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from models.layer_norm import LayerNorm

@dataclass
class MoLConfig:
    n_block: int = 1024
    n_embd: int = 768
    n_mix: int = 10
    n_tokens: int = 1
    bias: bool = True
    dropout: float = 0.0
    n_mlp_hidden: int = 128

class MoL(nn.Module):
    """
    Autoregressive over real-valued tokens.
    Head predicts K-component logistic mixture per step,
    with independent mixtures per token dimension.
    """
    def __init__(self, config: MoLConfig, backbone: nn.Module):
        super().__init__()
        self.n_block = config.n_block
        self.n_mix = config.n_mix
        self.n_tokens = config.n_tokens

        self.vte = nn.Sequential(
            nn.Linear(config.n_tokens, config.n_mlp_hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_mlp_hidden, config.n_embd, bias=config.bias),
        )

        # learned absolute positions
        self.wpe = nn.Embedding(config.n_block, config.n_embd)

        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # output head: 3K per token dimension (mix_logits, means, log_scales)
        self.lm_head = nn.Linear(config.n_embd, 3 * config.n_mix * config.n_tokens, bias=config.bias)

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor, state, targets: Tensor | None = None):
        """
        x:       [B, T, N] real-valued previous tokens
        targets: [B, T, N] real-valued next-token targets
        Returns:
          (mix_logits, mu, log_s) each [B,T,N,K], state, loss
        """
        device = x.device
        B, T, N = x.shape
        assert T <= self.n_block, f"seq len {T} > block size {self.n_block}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.vte(x)                                            # [B,T,n_embd]
        pos_emb = self.wpe(pos).unsqueeze(0).expand(B, T, -1)
        h_in = self.drop(tok_emb + pos_emb)

        h_out, state = self.backbone(h_in, state)
        h_out = self.ln_f(h_out)
        y = self.lm_head(h_out)                                          # [B,T,3*K*N]

        K = self.n_mix
        y = y.view(B, T, N, 3 * K)
        mix_logits, mu, log_s = torch.split(y, K, dim=-1)               # each [B,T,N,K]

        if targets is not None:
            logp = mixture_loglik(targets, mix_logits, mu, log_s)        # [B,T,N]
            loss = -logp.mean()
            return (mix_logits, mu, log_s), state, loss
        else:
            mix_logits = mix_logits[:, [-1], :, :]                       # [B,1,N,K]
            mu         = mu[:,       [-1], :, :]
            log_s      = log_s[:,    [-1], :, :]
            return (mix_logits, mu, log_s), state, None

    @torch.no_grad()
    def generate(self,
                 x_prefix: Tensor,
                 max_new_tokens: int,
                 state,
                 temperature: float = 1.0) -> Tensor:
        """
        Autoregressively sample continuous tokens.
        x_prefix: [B, T0, N] seed sequence
        Returns [B, T0 + max_new_tokens, N].
        """
        x = x_prefix
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.n_block else x[:, -self.n_block:]
            (mix_logits, mu, log_s), state, _ = self(x_cond, state, targets=None)
            # shapes [B,1,N,K]
            ml = mix_logits[:, -1, :, :]                               # [B,N,K]
            pi = F.softmax(ml, dim=-1)                                 # [B,N,K]

            # sample mixture indices per dimension
            B, N, K = pi.shape
            k = torch.distributions.Categorical(pi.view(-1, K)).sample()  # [B*N]
            k_onehot = F.one_hot(k, num_classes=K).float().view(B, N, K)

            # select component params
            mu_k    = (mu[:, -1, :, :]    * k_onehot).sum(dim=-1)      # [B,N]
            log_s_k = (log_s[:, -1, :, :] * k_onehot).sum(dim=-1)      # [B,N]

            # scale by temperature: convert to actual scale, multiply, sample directly
            s_k = (F.softplus(log_s_k) + 1e-8) * temperature
            u = torch.rand_like(mu_k).clamp_(1e-6, 1 - 1e-6)
            x_next = mu_k + s_k * (torch.log(u) - torch.log1p(-u))    # [B,N]
            x = torch.cat([x, x_next.unsqueeze(1)], dim=1)            # [B,T+1,N]

        return x

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()

# ---------- utilities ----------

def logistic_logpdf(x: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
    s = F.softplus(log_s) + 1e-8
    z = (x - mu) / s
    return -torch.log(s) + F.logsigmoid(z) + F.logsigmoid(-z)

def mixture_loglik(x: Tensor, mix_logits: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
    log_pi = F.log_softmax(mix_logits, dim=-1)                      # [...,K]
    x_exp = x.unsqueeze(-1)                                          # [...,1]
    log_comp = logistic_logpdf(x_exp, mu, log_s)                    # [...,K]
    return torch.logsumexp(log_pi + log_comp, dim=-1)               # [...]

def sample_logistic(mu: Tensor, log_s: Tensor) -> Tensor:
    u = torch.rand_like(mu).clamp_(1e-6, 1 - 1e-6)
    s = F.softplus(log_s) + 1e-8
    return mu + s * (torch.log(u) - torch.log1p(-u))
