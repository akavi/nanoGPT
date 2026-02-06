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
    Autoregressive over real-valued tokens x_t \in (-inf, inf).
    Head predicts K-component logistic mixture per step.
    """
    def __init__(self, config: MoLConfig, backbone: nn.Module):
        super().__init__()
        self.n_block = config.n_block
        self.n_mix = config.n_mix
        self.n_tokens = config.n_tokens

        # Chagpt claims this is better than a linear embedding
        # idk tho
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

        # output head: n_tokens * [mix_logits K | means K | log_scales K] = n_tokens * 3K per step
        self.lm_head = nn.Linear(config.n_embd, config.n_tokens * 3 * config.n_mix, bias=config.bias)

        self.scale_0 = nn.Parameter(torch.tensor(1.0)) 
        self.scale_rest = nn.Parameter(torch.tensor(1.0)) 

        self.apply(self._init_weights)
        # print parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        y_obs: torch.Tensor,
        state,
        targets: torch.Tensor | None = None,
    ):
        """
        y_obs:    [B, T, n_tokens] observed *bounded* tokens in (-1,1).
        targets:  if provided, [B, T, n_tokens] bounded y in (-1,1).
        """
        device = y_obs.device
        B, T, N = y_obs.shape
        assert T <= self.n_block
        assert N == self.n_tokens

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.vte(y_obs)                                      # [B,T,n_embd]
        pos_emb = self.wpe(pos).unsqueeze(0).expand(B, T, -1)
        h_in = self.drop(tok_emb + pos_emb)

        h_out, state = self.backbone(h_in, state)
        h_out = self.ln_f(h_out)
        y_params = self.lm_head(h_out)                                 # [B,T,N*3K]
        y_params = y_params.view(B, T, self.n_tokens, 3 * self.n_mix)  # [B,T,N,3K]
        mix_logits, mu, log_s = _pack_params(self.n_mix, y_params)     # each [B,T,N,K]

        if targets is not None:
            assert targets.shape == (B, T, self.n_tokens), \
                f"expected targets [B,T,{self.n_tokens}], got {tuple(targets.shape)}"

            logp = mixture_loglik(targets, mix_logits, mu, log_s)       # [B,T,N]
            loss = (-logp).mean()
            return (mix_logits, mu, log_s), state, loss
        else:
            return (
                mix_logits[:, [-1], :, :],
                mu[:,       [-1], :, :],
                log_s[:,    [-1], :, :],
            ), state, None


    @torch.no_grad()
    def generate(self, y_prefix: torch.Tensor, max_new_tokens: int, state,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        y_prefix: [B, T0, n_tokens] in (-1,1).
        Returns:  [B, T0+max_new_tokens, n_tokens] in (-1,1).
        """
        y = y_prefix
        for _ in range(max_new_tokens):
            y_cond = y if y.size(1) <= self.n_block else y[:, -self.n_block:]
            (mix_logits, mu, log_s), state, _ = self(y_cond, state, targets=None)

            # mix_logits/mu/log_s: [B, 1, N, K] -> take last step -> [B, N, K]
            pi = torch.softmax(mix_logits[:, -1, :, :], dim=-1)          # [B,N,K]
            k = torch.distributions.Categorical(pi).sample()             # [B,N]
            onehot = F.one_hot(k, num_classes=pi.size(-1)).float()       # [B,N,K]

            mu_k    = (mu[:, -1, :, :]    * onehot).sum(-1)              # [B,N]
            log_s_k = (log_s[:, -1, :, :] * onehot).sum(-1)              # [B,N]

            s_k = F.softplus(log_s_k) + 1e-8
            if temperature != 1.0:
                s_k = s_k * temperature

            u = torch.rand_like(mu_k).clamp_(1e-6, 1 - 1e-6)
            z_next = mu_k + s_k * (torch.log(u) - torch.log1p(-u))      # [B,N]

            y_next = z_next                                                # [B,N]
            y = torch.cat([y, y_next.unsqueeze(1)], dim=1)               # [B,T+1,N]
        return y

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
    log_pi = F.log_softmax(mix_logits, dim=-1)                      # [B,T,K]
    # expand x to [B,T,1] for broadcasting against [B,T,K]
    x_exp = x.unsqueeze(-1)
    log_comp = logistic_logpdf(x_exp, mu, log_s)                    # [B,T,K]
    return torch.logsumexp(log_pi + log_comp, dim=-1)               # [B,T]

def sample_logistic(mu: Tensor, log_s: Tensor) -> Tensor:
    u = torch.rand_like(mu).clamp_(1e-6, 1 - 1e-6)
    s = F.softplus(log_s) + 1e-8
    return mu + s * (torch.log(u) - torch.log1p(-u))  # logit(u)

def _pack_params(n_mix: int, y: Tensor):
    """
    y: [..., 3K] -> mix_logits [..., K], mu [..., K], log_s [..., K]
    """
    K = n_mix
    assert y.shape[-1] == 3 * K
    mix_logits, mu, log_s = torch.split(y, K, dim=-1)
    return mix_logits, mu, log_s
