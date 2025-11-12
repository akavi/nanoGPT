
# dmol_autoregressive.py
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.layernorm import LayerNorm
from models.utils import mean_by_quarter, mean_by_mod4  

# ---------- Config ----------

@dataclass
class DMoLConfig:
    n_block: int = 1024
    n_vocab: int = 256         # discrete 8-bit tokens for feedback/embedding
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    K: int = 10                # mixture components

# ---------- Model ----------

class DMoLAutoregressive(nn.Module):
    """
    Single-channel (grayscale) DMoL wrapper around a sequential backbone.
    """
    def __init__(self, config: DMoLConfig, backbone: nn.Module):
        super().__init__()
        self.config = config
        self.n_block = config.n_block
        self.K = config.K

        self.wte = nn.Embedding(config.n_vocab, config.n_embd)
        self.wpe = nn.Embedding(config.n_block, config.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, elementwise_affine=config.bias)

        # Head outputs: for grayscale, K*(pi, mean, log_scale) = 3K
        self.head = nn.Linear(config.n_embd, 3 * config.K, bias=True)

        self.apply(self._init_weights)

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")


    def forward(self, idx: Tensor, state, targets: Tensor | None = None, ignore_index: int = -1):
        """
        idx: [B,T] int tokens in [0,255] (previous pixel values)
        targets: [B,T] int tokens in [0,255] (next pixel values), or -1 at masked positions
        """
        device = idx.device
        B, T = idx.shape
        assert T <= self.n_block, f"len {T} > block {self.n_block}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok = self.wte(idx)             # [B,T,C]
        pos_emb = self.wpe(pos)         # [T,C]
        x = self.drop(tok + pos_emb)    # [B,T,C]

        x, state = self.backbone(x, state)

        x = self.ln_f(x)                # [B,T,C]
        head_out = self.head(x)         # [B,T,3K]
        pi_logits, means, log_scales = _split_head(head_out, self.K)

        loss = None
        loss_tok = None
        if targets is not None:
            # log prob per position
            logp = dmol_log_prob(targets, pi_logits, means, log_scales, ignore_index=ignore_index)  # [B,T]
            # negative log-likelihood
            loss_tok = -logp  # [B,T]

            # Optional stream weighting (your mean_by_* utils expect per-token loss)
            # Example: uniform average over valid tokens
            valid = (targets != ignore_index).float()
            denom = valid.sum().clamp_min(1.0)
            loss = (loss_tok * valid).sum() / denom

        if loss_tok is not None:
            print("mean by quarters", mean_by_quarter(loss_tok))
            print("mean by mod4", mean_by_mod4(loss_tok))

        # Return the raw head params to allow custom evaluation/sampling
        return (pi_logits, means, log_scales), state, loss

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,                # [B,t0] seed tokens in [0,255]
        max_new_tokens: int,
        state,
        temperature: float = 1.0,   # applied to mixture weights and scales
        top_k_components: int | None = None,  # prune mixture components at sampling
    ):
        self.eval()
        B = idx.size(0)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.n_block else idx[:, -self.n_block:]
            (pi_logits, means, log_scales), state, _ = self(idx_cond, state, targets=None)
            # take last time step
            pi = pi_logits[:, -1, :]                    # [B,K]
            mu = means[:, -1, :]                        # [B,K]
            ls = log_scales[:, -1, :]                   # [B,K]

            # temperature: soften mixture and widen scales
            if temperature != 1.0:
                pi = pi / temperature
                ls = ls + math.log(temperature)

            # optionally restrict to top-k mixture components
            if top_k_components is not None and top_k_components < pi.size(-1):
                v, ix = torch.topk(pi, top_k_components, dim=-1)
                mask = torch.full_like(pi, float('-inf'))
                pi_masked = mask.scatter(-1, ix, v)
            else:
                pi_masked = pi

            mix = torch.distributions.Categorical(logits=pi_masked).sample()  # [B]
            # select parameters per batch
            mu_sel = mu.gather(1, mix.unsqueeze(1)).squeeze(1)          # [B]
            s_sel = F.softplus(ls.gather(1, mix.unsqueeze(1)).squeeze(1)) + 1e-5  # [B]

            # sample from logistic via inverse CDF
            u = torch.rand(B, device=idx.device).clamp_(1e-6, 1-1e-6)
            x = mu_sel + s_sel * (torch.log(u) - torch.log1p(-u))       # continuous in [-∞,∞] but model learns [-1,1]

            # map back to byte, round+clamp
            x_byte = torch.clamp(torch.round((x + 1.0) * 0*_

    def flops_per_token(self):
        return self.backbone.flops_per_token()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

# ---------- Utilities: DMoL discretized likelihood ----------
def _split_head(h: Tensor, K: int):
    """
    h: [B,T,3K] -> pi_logits, means, log_scales each [B,T,K]
    """
    pi_logits, means, log_scales = torch.split(h, K, dim=-1)
    return pi_logits, means, log_scales

def _logistic_cdf(x: Tensor) -> Tensor:
    # sigmoid with clamp for stability
    return torch.sigmoid(x.clamp(min=-30.0, max=30.0))

def _log_sum_exp(a: Tensor, dim=-1) -> Tensor:
    m = a.max(dim=dim, keepdim=True).values
    return (a - m).exp().sum(dim=dim, keepdim=True).log() + m

def dmol_log_prob(
    x_int: Tensor,         # [B,T] uint8-like ints in [0,255] (or -1 for ignore)
    pi_logits: Tensor,     # [B,T,K]
    means: Tensor,         # [B,T,K]
    log_scales: Tensor,    # [B,T,K], unconstrained; softplus used internally
    ignore_index: int = -1,
) -> Tensor:
    """
    Return per-position log-prob (B,T) for 8-bit discrete x using CDF bin-masses.
    x is mapped to [-1,1] with bin width Δ=2/255. Edge bins handled as in PixelCNN++.
    """
    B, T, K = pi_logits.shape
    # mask ignored
    valid = (x_int != ignore_index)
    x_int = x_int.clamp(min=0)  # safe when ignored
    # map to continuous [-1,1]
    x = (x_int.float() / 255.0) * 2.0 - 1.0  # [B,T]
    x = x.unsqueeze(-1)  # [B,T,1]

    # positive scales
    s = F.softplus(log_scales) + 1e-5  # [B,T,K]

    # bin half width in [-1,1] space
    delta = 2.0 / 255.0

    # CDF at bin edges
    inv_s = 1.0 / s
    cdf_hi = _logistic_cdf((x + 0.5 * delta - means) * inv_s)
    cdf_lo = _logistic_cdf((x - 0.5 * delta - means) * inv_s)

    # exact bin mass; edge correction for x at endpoints
    # left edge (v==0): mass = cdf(x+Δ/2)
    # right edge (v==255): mass = 1 - cdf(x-Δ/2)
    v0 = (x_int == 0).unsqueeze(-1)
    v255 = (x_int == 255).unsqueeze(-1)

    bin_mass = torch.where(
        v0, cdf_hi,
        torch.where(v255, 1.0 - cdf_lo, (cdf_hi - cdf_lo).clamp_min(1e-12))
    )  # [B,T,K]

    log_mix = F.log_softmax(pi_logits, dim=-1)  # [B,T,K]
    log_probs = _log_sum_exp(torch.log(bin_mass.clamp_min(1e-12)) + log_mix, dim=-1).squeeze(-1)  # [B,T]

    # set ignored positions to 0 so downstream sums can mask them
    log_probs = torch.where(valid, log_probs, torch.zeros_like(log_probs))
    return log_probs  # [B,T]
