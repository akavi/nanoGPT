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

        self.scale_0 = nn.Parameter(torch.tensor(1.0)) 
        self.scale_rest = nn.Parameter(torch.tensor(1.0)) 

        self.apply(self._init_weights)
        # print parameter count
        n_params = sum(p.numel() for p in self.parameters())

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
        y_obs:    [B,T] observed *bounded* tokens in (-1,1).
        targets:  if provided, [B,T,2] where:
                  - targets[..., 0] are bounded y in (-1,1)
                  - targets[..., 1] are frequencies (>= 0), used to reweight loss
        """
        device = y_obs.device
        B, T = y_obs.shape
        assert T <= self.n_block

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok_emb = self.vte(y_obs.unsqueeze(-1))
        pos_emb = self.wpe(pos).unsqueeze(0).expand(B, T, -1)
        h_in = self.drop(tok_emb + pos_emb)

        h_out, state = self.backbone(h_in, state)
        h_out = self.ln_f(h_out)
        y_params = self.lm_head(h_out)  # [B,T,3K]
        mix_logits, mu, log_s = _pack_params(self.n_mix, y_params)

        if targets is not None:
            # 1) accept targets as [B,T,2]
            assert targets.ndim == 3 and targets.shape[:2] == (B, T) and targets.shape[2] == 2, \
                f"expected targets [B,T,2], got {tuple(targets.shape)}"

            # 2) compute loss using targets[:,:,0]
            y_targets = targets[:, :, 0]  # [B,T]
            # (Note: .squeeze() here can silently drop B or T when they equal 1; avoid unless you truly want that.)
            # y_targets = targets[:, :, 0].squeeze()

            # map bounded targets y -> latent z via atanh, and add log|det J|
            z_targets = atanh_safe(y_targets)  # [B,T]
            logp_z = mixture_loglik(z_targets, mix_logits, mu, log_s)  # [B,T]
            log_det = log_abs_det_jacobian_tanh_from_y(y_targets)      # [B,T]
            logp_y = logp_z + log_det
            loss_tok = -logp_y  # [B,T]

            # 3) reweight by inverse log frequency: 1 / log(1 + freq)
            freq = targets[:, :, 1]  # [B,T]
            weight = 1.0 / (1.0 + freq)
            loss_tok = loss_tok * weight

            loss = loss_tok.mean()
            return (mix_logits, mu, log_s), state, loss
        else:
            # inference-time: return only last-step params
            return (
                mix_logits[:, [-1], :],
                mu[:,       [-1], :],
                log_s[:,    [-1], :],
            ), state, None


    @torch.no_grad()
    def generate(self, y_prefix: torch.Tensor, max_new_tokens: int, state,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        y_prefix: [B,T0] in (-1,1). We sample latent z from the MoL, then squash with tanh to (-1,1).
        """
        y = y_prefix
        for _ in range(max_new_tokens):
            y_cond = y if y.size(1) <= self.n_block else y[:, -self.n_block:]
            (mix_logits, mu, log_s), state, _ = self(y_cond, state, targets=None)

            # mixture weights (optionally sharpen/flatten)
            pi = torch.softmax(mix_logits[:, -1, :], dim=-1)             # [B,K]
            k = torch.distributions.Categorical(pi).sample()             # [B]
            onehot = F.one_hot(k, num_classes=pi.size(-1)).float()

            mu_k    = (mu[:, -1, :]    * onehot).sum(-1)                 # [B]
            log_s_k = (log_s[:, -1, :] * onehot).sum(-1)                 # [B]

            # temperature as scale multiplier in latent space
            s_k = F.softplus(log_s_k) + 1e-8
            if temperature != 1.0:
                s_k = s_k * temperature

            # sample latent z from Logistic(mu_k, s_k)
            u = torch.rand_like(mu_k).clamp_(1e-6, 1 - 1e-6)
            z_next = mu_k + s_k * (torch.log(u) - torch.log1p(-u))       # [B]

            # squash to (-1,1)
            y_next = torch.tanh(z_next)
            y = torch.cat([y, y_next.unsqueeze(1)], dim=1)
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

def atanh_safe(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # keep y strictly inside (-1, 1) to avoid infinities
    y = y.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(y) - torch.log1p(-y))  # atanh(y)

def log_abs_det_jacobian_tanh_from_y(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # z = atanh(y); dz/dy = 1 / (1 - y^2)
    return -torch.log1p(-y.mul(y)).clamp_min(eps)  # -log(1 - y^2)

def _pack_params(n_mix: int, y: Tensor):
    """
    y: [B,T,3K] -> mix_logits [B,T,K], mu [B,T,K], log_s [B,T,K]
    """
    B, T, C = y.shape
    K = n_mix
    assert C == 3 * K
    mix_logits, mu, log_s = torch.split(y, K, dim=-1)
    return mix_logits, mu, log_s
