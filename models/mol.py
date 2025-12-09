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
    Autoregressive over real-valued tokens x_t.

    x_cont: [B, T, D]
      - feature 0: value (the thing we are modeling)
      - feature 1: frequency (used to scale mu), if present
      - feature 2+: ignored for now
    """
    def __init__(self, config: MoLConfig, backbone: nn.Module):
        super().__init__()
        self.n_block = config.n_block
        self.n_mix = config.n_mix

        # Value-to-embedding MLP (takes only the value, not frequency)
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

        # Learnable affine for frequency scaling of mu: mu <- mu * (w_m * freq + w_b)
        self.w_m = nn.Parameter(torch.tensor(float(2**16)))  # 65536.0
        self.w_b = nn.Parameter(torch.tensor(0.0))

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

    def forward(self, x_cont: Tensor, state, targets: Tensor | None = None):
        """
        x_cont: [B, T, D]
            x_cont[..., 0] = value
            x_cont[..., 1] = frequency (if present)
        targets: [B, T] real-valued next-token targets

        Returns:
          (mix_logits, mu, log_s), state, (loss or None)
        """
        assert x_cont.ndim == 3, f"x_cont must be [B,T,D], got {x_cont.shape}"
        B, T, D = x_cont.shape
        assert D >= 1, f"expected at least 1 feature, got {D}"
        assert T <= self.n_block, f"seq len {T} > block size {self.n_block}"

        device = x_cont.device

        # Split value vs frequency
        x_val = x_cont[..., 0]             # [B, T]
        x_freq = x_cont[..., 1]

        pos = torch.arange(0, T, dtype=torch.long, device=device)     # [T]

        # value embedding ONLY (first feature)
        tok_emb = self.vte(x_val.unsqueeze(-1))                       # [B, T, n_embd]
        pos_emb = self.wpe(pos).unsqueeze(0).expand(B, T, -1)        # [B, T, n_embd]
        h_in = self.drop(tok_emb + pos_emb)

        h_out, state = self.backbone(h_in, state)                    # [B, T, n_embd], state
        h_out = self.ln_f(h_out)
        y = self.lm_head(h_out)                                      # [B, T, 3K]
        mix_logits, mu, log_s = torch.split(y, self.n_mix, dim=-1)   # all [B, T, K]

        # frequency-dependent scaling of mu
        # x_freq: [B, T]
        scale = self.w_m * x_freq + self.w_b                    # [B, T]
        mu = mu * scale.unsqueeze(-1)                           # [B, T, K]

        print(
            f"logits:\n {mix_logits[0,0,:].tolist()}\n, "
            f"mu:\n {mu[0,0,:].tolist()}\n, "
            f"log_s:\n {log_s[0,0,:].tolist()}\n"
        )

        if targets is not None:
            # continuous NLL of mixture of logistics
            logp = mixture_loglik(targets, mix_logits, mu, log_s)    # [B, T]
            loss_tok = -logp                                         # per-token loss
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
                 schedule: Tensor,
                 state,
                 temperature: float = 1.0) -> Tensor:
        """
        Autoregressively sample continuous tokens.

        x_prefix: [B, T0]
            - value channel only (same scale as training values)
        schedule: [B, T0 + max_new_tokens, D_extras]
            - D_extras = D - 1, i.e. all non-value features (e.g. frequency)
            - schedule[:, t, :] gives the extra features for position t
        state: backbone state
        Returns:
            x: [B, T0 + max_new_tokens] (values only)
        """
        assert x_prefix.ndim == 2, f"x_prefix must be [B,T0], got {x_prefix.shape}"
        assert schedule.ndim == 3, f"schedule must be [B,T_total,D_extras], got {schedule.shape}"

        x = x_prefix
        B, T0 = x.shape
        B_s, T_total, D_extras = schedule.shape
        assert B_s == B, f"batch mismatch: x_prefix B={B}, schedule B={B_s}"
        assert T_total >= T0, f"T_total={T_total} must be >= T0={T0}"

        schedule = schedule.to(x.device)
        max_new_tokens = T_total - T0

        for _ in range(max_new_tokens):
            T_current = x.size(1)

            # context window
            if T_current <= self.n_block:
                start_idx = 0
            else:
                start_idx = T_current - self.n_block
            end_idx = T_current

            # slice values and extra features to the same window
            x_vals_cond = x[:, start_idx:end_idx]                         # [B, T_cond]
            feat_cond   = schedule[:, start_idx:end_idx, :]               # [B, T_cond, D_extras]

            # build x_cont: [value | extra_features]
            x_cont = torch.cat(
                [x_vals_cond.unsqueeze(-1), feat_cond], dim=-1
            )                                                              # [B, T_cond, 1 + D_extras]

            (mix_logits, mu, log_s), state, _ = self(x_cont, state, targets=None)
            # mix_logits, mu, log_s are [B,1,K] here

            mix_logits_step = mix_logits[:, -1, :] / 1.0                  # [B,K]
            pi = F.softmax(mix_logits_step, dim=-1)                       # [B,K]

            # sample mixture indices
            k = torch.distributions.Categorical(pi).sample()              # [B]
            k_onehot = F.one_hot(k, num_classes=pi.size(-1)).float()      # [B,K]

            # select component params
            mu_k    = (mu[:, -1, :]    * k_onehot).sum(dim=-1)            # [B]
            log_s_k = (log_s[:, -1, :] * k_onehot).sum(dim=-1)            # [B]

            # temperature on scale
            if temperature != 1.0:
                # equivalent to multiplying s by temperature
                log_s_k = torch.log(F.softplus(log_s_k) * temperature + 1e-8)

            x_next = sample_logistic(mu_k, log_s_k)                       # [B]
            x = torch.cat([x, x_next.unsqueeze(1)], dim=1)                # [B, T_current+1]

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
    log_pi = F.log_softmax(mix_logits, dim=-1)                      # [B,T,K]
    # expand x to [B,T,1] for broadcasting against [B,T,K]
    x_exp = x.unsqueeze(-1)
    log_comp = logistic_logpdf(x_exp, mu, log_s)                    # [B,T,K]
    return torch.logsumexp(log_pi + log_comp, dim=-1)               # [B,T]

def sample_logistic(mu: Tensor, log_s: Tensor) -> Tensor:
    u = torch.rand_like(mu).clamp_(1e-6, 1 - 1e-6)
    s = F.softplus(log_s) + 1e-8
    return mu + s * (torch.log(u) - torch.log1p(-u))  # logit(u)
