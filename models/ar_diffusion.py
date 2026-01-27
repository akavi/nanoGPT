import math
from dataclasses import dataclass
from typing import Literal, Optional, Any
from models.model import MLP
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor, nn
from models.utils import mean_by_quarter, mean_by_mod4
from models.layer_norm import LayerNorm

@dataclass
class ArDiffusionConfig:
    n_block: int
    n_vocab: int
    n_embd: int
    n_step: int
    dropout: float
    mode: Literal["train"] | Literal["sample"]
    latent_loss_scale: float
    device: str
    gamma: float
    snr_eps: float = 0.1

class ArDiffusion(nn.Module):
    """
    Autoregressive model with diagonal diffusion ladders.
    
    At each sequence position i, the model sees a "ladder" of S tokens at 
    decreasing noise levels:
        input[i] = [tok_{i-S+1} @ step S-1 (cleanest),
                    tok_{i-S+2} @ step S-2,
                    ...,
                    tok_i @ step 0 (noisiest)]
    
    The model outputs the *next* ladder:
        output[i] = [tok_{i-S+2} @ step S-1,
                     tok_{i-S+3} @ step S-2,
                     ...,
                     tok_{i+1} @ step 0]
    
    Each token thus "ascends" the ladder across sequence positions: appearing 
    first at step 0 (noisy), then at step 1 (cleaner) in the next position, 
    etc., until reaching step S-1 (clean) S-1 positions later.
    
    Losses:
        - CE on tok_{i+1} from the cleanest output slot (step S-1)
        - MSE between output ladder and next position's input ladder, 
          encouraging the model to predict the partially-denoised versions
          that will appear in subsequent inputs
    
    At inference, the model fills each rung by re-using its own outputs from
    previous positions, so rungs below S-1 contain model predictions rather
    than noised ground truth.
    """
    def __init__(self, config, backbone: nn.Module):
        super().__init__()
        assert config.n_vocab is not None
        assert config.n_block is not None
        self.mode = config.mode
        self.n_block = config.n_block
        self.n_vocab = config.n_vocab
        self.n_step = config.n_step
        self.latent_loss_scale = config.latent_loss_scale
        self.gamma = config.gamma
        self.snr_eps = config.snr_eps
        self.n_embd = config.n_embd
        self.device = config.device

        self.n_embd_per_step = config.n_embd // config.n_step
        self.wte = nn.Embedding(config.n_vocab, self.n_embd_per_step)
        self.wpe = nn.Embedding(config.n_block + config.n_step - 1, self.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.in_norm = nn.Identity(self.n_step, self.n_embd_per_step)
        self.out_norm = SubLatentLayerNorm(self.n_step, self.n_embd_per_step)

        self.pre_lm_head = MLP(self.n_embd_per_step)
        self.lm_projection = nn.Linear(self.n_embd_per_step, config.n_vocab, bias=False)
        self.wte.weight = self.lm_projection.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _alpha_schedule(self, device: torch.device) -> Tensor:
        """
        SNR/variance-space ladder: alpha_s is the *signal variance fraction*.
        alpha in (0,1], increasing with s, with alpha[S-1]=1 (clean).
        """
        S = self.n_step
        # If you want exactly your old "no pure noise" semantics:
        alpha = torch.linspace(1.0 / S, 1.0, steps=S, device=device)  # shape (S,)
        # Optional shaping knob using gamma (keep gamma=0 for linear):
        if getattr(self, "gamma", 0.0) != 0.0:
            # This is a mild heuristic; feel free to replace with logSNR shaping.
            alpha = alpha ** (1.0 - self.gamma)
            alpha[-1] = 1.0
        return alpha

    def _prep_backbone_inputs(
        self,
        toks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        LN-space noising with per-AR-position Gaussian increments.

        We first form the clean ladder embeddings, then map them into "LN space"
        (zero-mean, unit-variance per vector, no affine). We then construct a
        correlated noise-direction field via per-position increments and window sums,
        orthogonalize that direction to the clean vector, and finally mix by an
        SNR/variance-space schedule in LN space:

            u = LN_noaffine(clean_emb)
            v = LN_noaffine( proj_orth( eps, u ) )
            x_raw[..., s, :] = sqrt(alpha_s) * u + sqrt(1-alpha_s) * v

        Returns:
          x_in      : (B, L, S, E) ladder inputs (passed through self.in_norm)
          train_mask: (1, L, S, 1)
          gen_mask  : (1, L, S, 1)
          x_raw     : (B, L, S, E) pre-in_norm ladder (already ~LN-space)
        """
        device = toks.device
        B, t = toks.size()
        assert t <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"

        S = self.n_step
        E = self.n_embd_per_step

        side_negative_mask = torch.zeros(1, S - 1, S, 1, device=device)
        side_positive_mask = torch.ones(1, S - 1, S, 1, device=device)
        main_positive_mask = torch.ones(1, t, S, 1, device=device)

        train_mask = tilt(
            torch.concat([side_negative_mask, main_positive_mask, side_negative_mask], dim=1),
            tilt_dim=2,
            content_dim=1,
        )  # (1, L, S, 1)

        gen_mask = tilt(
            torch.concat([side_positive_mask, main_positive_mask, side_negative_mask], dim=1),
            tilt_dim=2,
            content_dim=1,
        )  # (1, L, S, 1)

        # ---- clean embeddings, tilted ----
        emb_toks = self.wte(toks)  # (B, t, E)
        exp_emb = emb_toks.unsqueeze(-2).expand(B, t, S, E)  # (B, t, S, E)
        exp_emb = self.in_norm(exp_emb)

        left_pad  = self.in_norm(torch.ones(B, S - 1, S, E, device=device))
        right_pad = self.in_norm(torch.ones(B, S - 1, S, E, device=device))
        cat_emb = torch.cat([left_pad, exp_emb, right_pad], dim=1)          # (B, t+2(S-1), S, E)
        emb_tilted = tilt(cat_emb, tilt_dim=2, content_dim=1)               # (B, L, S, E)

        Ln = t + S - 1
        # Make noise isotropic in LN space, orthogonal to clean embedding
        noise = torch.randn(B, Ln, S, E, device=device)
        # Project out the component parallel to emb_tilted
        dot = (noise * emb_tilted).sum(dim=-1, keepdim=True)
        norm_sq = (emb_tilted * emb_tilted).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        noise = noise - (dot / norm_sq) * emb_tilted
        # Normalize to unit variance in LN space
        noise = self.in_norm(noise)

        alpha = self._alpha_schedule(device)  # (S,)
        schedule = alpha.view(1, 1, S, 1)  # (1, 1, S, 1) for broadcasting
        noised_schedule = (schedule + (1 - schedule) * (1 + 0.2 * torch.randn(1, device=device))).clamp(0, 1)
        print(f"noised_schedule: {noised_schedule.flatten().tolist()}")

        # Mix emb_tilted and noise via variance-preserving interpolation (slerp in variance space)
        # x = sqrt(alpha) * clean + sqrt(1 - alpha) * noise
        x_in = torch.sqrt(noised_schedule) * emb_tilted + torch.sqrt(1 - noised_schedule) * noise
        return x_in, emb_tilted, schedule, train_mask, gen_mask

    def forward(self, toks, state, targets=None):
        diffusion_state, backbone_state = state

        x_in, x_raw, schedule, train_mask, gen_mask = self._prep_backbone_inputs(toks)

        (B, T) = toks.size()
        L = x_in.shape[1]
        S = self.n_step
        V = self.n_vocab

        if self.mode == "train":
            y, new_backbone_state = self._one_step(x_in, backbone_state)
            tok_logits = self.lm_head(y)  # (B, L, S, V)
            new_diff_state = y

            # ---- denoising ratio diagnostics (NORMED space) ----
            with torch.no_grad():
                if self.n_step > 1:
                    input_s = torch.randn_like(x_raw[:, 1:, 0, :])  # (B, L-1, E)
                    target = x_raw[:, 1:, 0, :]  # (B, L-1, E)
                    output_cleaner = y[:, :-1, 1, :] # (B, L-1, E)
                    input_to_gt = ((input_s - target) ** 2).mean()
                    output_to_gt = ((output_cleaner - target) ** 2).mean()
                    print(f"NOISE->0: input_to_gt={input_to_gt:.4f}, output_to_gt={output_to_gt:.4f}, ratio={output_to_gt/input_to_gt:.4f}")
                for s in range(S - 1):
                    input_s = x_in[:, :-1, s, :]         # (B, L-1, E)
                    target = x_raw[:, :-1, s, :]  # (B, L-1, E)
                    output_cleaner = y[:, :-1, s + 1, :] # (B, L-1, E)

                    input_to_gt = ((input_s - target) ** 2).mean()
                    output_to_gt = ((output_cleaner - target) ** 2).mean()
                    print(f"step {s}->{s+1}: input_to_gt={input_to_gt:.4f}, output_to_gt={output_to_gt:.4f}, ratio={output_to_gt/input_to_gt:.4f}")

            # ---- targets / losses (unchanged) ----
            side_target_mask = torch.zeros(B, S - 1, S, dtype=toks.dtype, device=toks.device)
            targets = toks.unsqueeze(-1).expand(B, T, S)  # (B, T, S)
            targets = tilt(
                torch.concat([side_target_mask, targets, side_target_mask], dim=1),
                tilt_dim=2,
                content_dim=1,
            )  # (B, L, S)

            Ln = L - 1
            ce_loss_per = F.cross_entropy(
                tok_logits[:, :-1, :, :].reshape(B * Ln * S, V),
                targets[:, 1:, :].reshape(B * Ln * S),
                reduction="none",
            ).reshape(B, Ln, S)

            latent_loss_per = _latent_mse(
                pred=y[:, :-1, :, :],
                target=x_in[:, 1:, :, :].detach(),
                real_mask=train_mask[:, 1:, :, :],
            )

            # Weight CE losses by SNR with offset (cleaner steps get higher weight)
            alpha = schedule[:, :, :, 0]  # (1, 1, S)
            snr = (alpha + self.snr_eps) / (1 - alpha + self.snr_eps)  # (1, 1, S)
            mask = train_mask[:, 1:, :, 0]  # (1, Ln, S)
            ce_loss = (ce_loss_per * snr * mask).sum() / (B * (snr * mask).sum()).clamp(min=1e-8)
            latent_loss = latent_loss_per
            loss = ce_loss + self.latent_loss_scale * latent_loss

            # ---- gradient alignment diagnostic ----
            if self.latent_loss_scale > 0 and torch.is_grad_enabled():
                grad_ce = torch.autograd.grad(ce_loss, y, retain_graph=True)[0]
                grad_latent = torch.autograd.grad(latent_loss, y, retain_graph=True)[0]
                # Flatten and compute cosine similarity
                g_ce = grad_ce.flatten()
                g_lat = grad_latent.flatten()
                cos_sim = F.cosine_similarity(g_ce.unsqueeze(0), g_lat.unsqueeze(0)).item()
                # Also compute relative magnitudes
                ce_norm = g_ce.norm().item()
                latent_norm = g_lat.norm().item()
                print(f"grad alignment: cos_sim={cos_sim:.4f}, ce_norm={ce_norm:.4f}, latent_norm={latent_norm:.4f}, ratio={ce_norm/latent_norm:.4f}")


            return tok_logits, (new_diff_state, new_backbone_state), loss

        else:  # sample
            fill_length = toks.shape[1] + self.n_step - 1
            for idx in range(diffusion_state.shape[1] - 1, fill_length):
                m = train_mask[:, idx:idx+1, :, :].to(dtype=x_in.dtype)  # (1,1,S,1)
                x_in[:, idx:idx+1, :, :] = (
                    m * x_in[:, idx:idx+1, :, :] + (1.0 - m) * diffusion_state[:, idx:idx+1, :, :]
                )
                y, backbone_state = self._one_step(
                    x_in[:, :idx+1, :, :],
                    backbone_state,
                    pos_idx=idx,
                )
                diffusion_state = torch.concat([diffusion_state, y[:, -1:, :, :]], dim=1)

            tok_logits = self.lm_head(y[:, :, -1, :])  # (B, T, V) from cleanest sublatent
            return tok_logits, (diffusion_state, backbone_state), None

    def lm_head(self, x: Tensor) -> Tensor:
        # x = self.pre_lm_head(x)  # (B,L,E)
        return self.lm_projection(x)  # (B,L,V)

    @torch.no_grad()
    def generate(self, tok, max_new_tokens, state, temperature=1.0, top_k=None):
        # implementation from 
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for idx in range(max_new_tokens):
            print(f"gen {idx}")
            # if the sequence context is growing too long we must crop it at n_block
            tok = tok if tok.size(1) <= self.n_block else tok[:, -self.n_block:]
            # forward the model to get the logits for the index in the sequence
            logits, state, _ = self(tok, state)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            tok = torch.cat((tok, x_next), dim=1)

        return tok
    
    
    def initial_state(self, batch_size):
        diffusion_state = torch.zeros(batch_size, 1, self.n_step, self.n_embd_per_step, device=self.device)
        return (diffusion_state, self.backbone.initial_state(batch_size))

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params


    def _one_step(self, x: Tensor, backbone_state, pos_idx=0):
        B, L, _, _ = x.shape
        x_flat = x.reshape(B, L, self.n_step * self.n_embd_per_step)
        pos = torch.arange(0, L, device=self.device)
        pos_emb = self.wpe(pos)  # (L, n_embd)
        x_flat = self.drop(pos_emb + x_flat)

        y_flat, new_backbone_state = self.backbone(x_flat, backbone_state)  # (B,L,n_embd)

        suffix_len = L - pos_idx
        y_flat = y_flat[:, -suffix_len:, :]

        y_pre = y_flat.view(B, L - pos_idx, self.n_step, self.n_embd_per_step)        # (B,L,S,E)
        y = self.out_norm(y_pre)
        return y, new_backbone_state


def tilt(
    a: torch.Tensor,
    *,
    tilt_dim: int = -2,     # “row” dim: offset increases with this index
    content_dim: int = -1,  # “col” dim: gets truncated
) -> torch.Tensor:
    """
    Returns a diagonally-“tilted” view of `a` without padding:
      out[row, :] = a[row, row : row+K]
    where K = keep if provided, else K = D - (T-1).

    Example (T=4, D=5): K = 5-(4-1)=2
      [[1,2,3,4,5],
       [6,7,8,9,10],
       [11,12,13,14,15],
       [16,17,18,19,20]]
    -> [[1,2],[7,8],[13,14],[19,20]]
    """
    nd = a.ndim
    tilt_dim = tilt_dim % nd
    content_dim = content_dim % nd
    if tilt_dim == content_dim:
        raise ValueError("tilt_dim and content_dim must be different")

    # Move chosen dims to the end: [...batch..., T, D]
    other = [d for d in range(nd) if d not in (tilt_dim, content_dim)]
    perm = other + [tilt_dim, content_dim]
    inv_perm = [0] * nd
    for i, p in enumerate(perm):
        inv_perm[p] = i

    ap = a.permute(perm)
    *batch, T, D = ap.shape
    device = ap.device

    K = (D - (T - 1))
    if (T - 1) + K > D:
        raise ValueError(f"Invalid keep={K}: last row would index past D (need (T-1)+keep <= D).")

    rows = torch.arange(T, device=device).view(T, 1)          # [T,1]
    offs = torch.arange(K, device=device).view(1, K)          # [1,K]
    start = (D - K) - rows          # (T,1)
    idx = start + offs              # (T,K)

    idx = idx.view((1,) * len(batch) + (T, K)).expand(*batch, T, K)
    out = ap.gather(-1, idx)

    return out.permute(inv_perm)

def _latent_mse(pred: Tensor, target: Tensor, real_mask: Tensor) -> Tensor:
    """
    Masked MSE over latent ladders.

    pred:      (B, L, S, E)
    target:    (B, L, S, E)
    real_mask: broadcastable to (B, L, S, E) (e.g. (1, L, S, 1) or (B, L, S, 1))
              1 = include, 0 = ignore

    Returns: scalar tensor (float)
    """

    m_exp = real_mask.expand_as(pred)
    se = (pred - target).pow(2) * m_exp
    return se.sum() / m_exp.sum()

class SubLatentLayerNorm(nn.Module):
    def __init__(self, S: int, E: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, S, E)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps)
