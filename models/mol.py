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
    bias: bool = True
    dropout: float = 0.0
    n_mlp_hidden: int = 128

class MoL(nn.Module):
    """
    Autoregressive over real-valued tokens x_t \in (-inf, inf).
    Head predicts K-component logistic mixture per step.
    """
    def __init__(self, config: MoLConfig, backbone: nn.Module):
        super().__init__()
        self.n_block = config.n_block
        self.n_mix = config.n_mix

        # Chagpt claims this is better than a linear embedding
        # idk tho
        self.vte = nn.Sequential(
            nn.Linear(1, config.n_mlp_hidden, bias=config.bias),
            nn.GELU(),
            nn.Linear(config.n_mlp_hidden, config.n_embd, bias=config.bias),
        )

        # learned absolute positions
        self.wpe = nn.Embedding(config.n_block, config.n_embd)

        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # output head: [mix_logits K | means K | log_scales K] = 3K per step
        self.lm_head = nn.Linear(config.n_embd, 3 * config.n_mix, bias=config.bias)

        self.apply(self._init_weights)
        # print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initial_state(self, batch_size: int):
        return self.backbone.initial_state(batch_size)

    def forward(self, x_cont: Tensor, state, targets: Tensor | None = None):
        """
        x_cont:  [B, T] real-valued previous tokens
        targets: [B, T] real-valued next-token targets (aligned with x_cont positions)
        Returns:
          params tuple for last step if targets is None (inference), else full-step loss.
        """
        device = x_cont.device
        B, T = x_cont.shape
        assert T <= self.n_block, f"seq len {T} > block size {self.n_block}"

        pos = torch.arange(0, T, dtype=torch.long, device=device)    # [T]
        # value embedding
        tok_emb = self.vte(x_cont.unsqueeze(-1))                     # [B,T,n_embd]
        pos_emb = self.wpe(pos).unsqueeze(0).expand(B, T, -1)        # [B,T,n_embd]
        h_in = self.drop(tok_emb + pos_emb)

        h_out, state = self.backbone(h_in, state)                    # [B,T,n_embd], state
        h_out = self.ln_f(h_out)
        y = self.lm_head(h_out)                                      # [B,T,3K]
        mix_logits, mu, log_s = torch.split(y, self.n_mix, dim=-1)

        if targets is not None:
            # continuous NLL of mixture of logistics
            logp = mixture_loglik(targets, mix_logits, mu, log_s)    # [B,T]
            loss_tok = -logp                                          # per-token loss
            loss = loss_tok.mean()
            return (mix_logits, mu, log_s), state, loss
        else:
            # inference: return only last-step params for efficiency
            mix_logits = mix_logits[:, [-1], :]                      # [B,1,K]
            mu         = mu[:,       [-1], :]                        # [B,1,K]
            log_s      = log_s[:,    [-1], :]                        # [B,1,K]
            return (mix_logits, mu, log_s), state, None

    @torch.no_grad()
    def generate(self,
                 x_prefix: Tensor,
                 max_new_tokens: int,
                 state,
                 temperature: float = 1.0) -> Tensor:
        """
        Autoregressively sample continuous tokens.
        x_prefix: [B, T0] seed sequence (can be empty T0=0 with a learned BOS if you add one)
        Returns concatenated sequence [B, T0 + max_new_tokens].
        Temperature scales *component scales*; 1.0 = nominal, >1.0 = more spread.
        """
        x = x_prefix
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.n_block else x[:, -self.n_block:]
            (mix_logits, mu, log_s), state, _ = self(x_cond, state, targets=None)
            # shapes [B,1,K]
            mix_logits = mix_logits[:, -1, :] / 1.0                   # you could divide by a "mixture temp" if desired
            pi = F.softmax(mix_logits, dim=-1)                        # [B,K]

            # sample mixture indices
            k = torch.distributions.Categorical(pi).sample()          # [B]
            k_onehot = F.one_hot(k, num_classes=pi.size(-1)).float()  # [B,K]

            # select component params
            mu_k    = (mu[:, -1, :]    * k_onehot).sum(dim=-1)        # [B]
            log_s_k = (log_s[:, -1, :] * k_onehot).sum(dim=-1)        # [B]

            # temperature on scale
            if temperature != 1.0:
                # equivalent to multiplying s by temperature
                log_s_k = torch.log(F.softplus(log_s_k) * temperature + 1e-8)

            x_next = sample_logistic(mu_k, log_s_k)                   # [B]
            x = torch.cat([x, x_next.unsqueeze(1)], dim=1)            # [B, T+1]

        return x

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def flops_per_token(self):
        return self.backbone.flops_per_token()

# ---------- utilities ----------

def logistic_logpdf(x: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
    s = F.softplus(log_s) + 1e-8
    z = (x - mu) / s
    return -torch.log(s) + F.logsigmoid(z) + F.logsigmoid(-z)

def mixture_loglik(x: Tensor, mix_logits: Tensor, mu: Tensor, log_s: Tensor) -> Tensor:
    log_pi = F.log_softmax(mix_logits, dim=-1)                      # [B,T,K]
    # expand x to [B,T,1] for broadcasting against [B,T,K]
    x_exp = x.unsqueeze(-1)
    log_comp = logistic_logpdf(x_exp, mu, log_s)                    # [B,T,K]
    return torch.logsumexp(log_pi + log_comp, dim=-1)               # [B,T]

def sample_logistic(mu: Tensor, log_s: Tensor) -> Tensor:
    u = torch.rand_like(mu).clamp_(1e-6, 1 - 1e-6)
    s = F.softplus(log_s) + 1e-8
    return mu + s * (torch.log(u) - torch.log1p(-u))  # logit(u)

