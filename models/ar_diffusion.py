from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypedDict, cast

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

# NOTE: These imports are used elsewhere in the repo; keep them to avoid breaking callers.
from models.utils import mean_by_quarter, mean_by_mod4  # noqa: F401
from models.layer_norm import LayerNorm  # noqa: F401


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


class _DiffusionState(TypedDict):
    # Current packed latent window x_pos to feed into the backbone for the next step.
    # Shape: (B, n_step, n_embd_per_step)
    x: Tensor
    # Number of tokens already consumed (i.e., number of backbone steps run).
    pos: int


class ArDiffusion(nn.Module):
    """
    Autoregressive diffusion-in-latent-space.

    Core training semantics (teacher forcing):
      - Build packed inputs x_j, j=0..T-1, each is a window of n_step sub-latents:
          x_j[s] = noisy embedding of token (j - s) at noise-level s
        where s=0 is cleanest, s=n_step-1 is noisiest. Out-of-range tokens are pure noise (X).
      - Backbone predicts the *next* packed latent:
          y_j ≈ x_{j+1}
      - Token logits at step j are produced from the cleanest sub-latent of y_j (s=0),
        and trained with standard next-token cross entropy vs toks[:, j+1].

    Sampling semantics:
      - Maintain diffusion_state.x = current packed latent x_pos.
      - For each newly-seen token tok[pos], overwrite the cleanest slot (s=0) in x_pos with its embedding,
        run the backbone for one step to obtain y_pos (≈ x_{pos+1}),
        emit logits from y_pos[s=0] (predicting next token),
        then set diffusion_state.x = y_pos and increment pos.
    """

    def __init__(self, config: ArDiffusionConfig, backbone: nn.Module):
        super().__init__()
        assert config.n_vocab is not None
        assert config.n_block is not None
        assert config.n_step is not None
        assert config.n_embd % config.n_step == 0, "n_embd must be divisible by n_step"

        self.mode = config.mode
        self.latent_loss_scale = config.latent_loss_scale
        self.device = config.device

        self.n_block = config.n_block
        self.n_step = config.n_step
        self.n_embd = config.n_embd
        self.n_embd_per_step = config.n_embd // config.n_step

        self.backbone = backbone
        self.wte = nn.Embedding(config.n_vocab, self.n_embd_per_step)
        self.wpe = nn.Embedding(config.n_block, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(self.n_embd_per_step, config.n_vocab, bias=False)

        # tie weights
        self.wte.weight = self.lm_head.weight

        # init weights (match style in categorical.py)
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _noise_weights(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Linear schedule: s=0 clean -> w=0, s=n_step-1 noisy -> w=1."""
        w = torch.linspace(0.0, 1.0, steps=self.n_step, device=device, dtype=dtype)
        return w.view(1, 1, self.n_step, 1)  # (1,1,n_step,1)

    def _prep_backbone_inputs(
        self,
        toks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Construct teacher-forced packed latents x_j for training.

        Returns:
          x_flat: (B, T, n_embd) packed inputs to the backbone
          x:      (B, T, n_step, n_embd_per_step) packed inputs (unflattened)
          real:   (T, n_step) boolean mask: True where x[j,s] corresponds to an in-range token (not X)
        """
        device = toks.device
        B, T = toks.shape
        assert T <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"
        if T > self.n_block:
            raise ValueError(f"Cannot forward sequence of length {T}, block size is only {self.n_block}")

        emb = self.wte(toks)  # (B,T,d)
        d = emb.shape[-1]
        w = self._noise_weights(device, emb.dtype)  # (1,1,S,1)

        # Shared per-token noise ε_i reused across s.
        eps = torch.randn(B, T, d, device=device, dtype=emb.dtype)
        emb_exp = emb.unsqueeze(2)  # (B,T,1,d)
        eps_exp = eps.unsqueeze(2)  # (B,T,1,d)
        noised = emb_exp * (1.0 - w) + eps_exp * w  # (B,T,S,d)

        # Left padding provides pure-noise X for out-of-range token indices.
        pad_len = self.n_step - 1
        pad = torch.randn(B, pad_len, self.n_step, d, device=device, dtype=emb.dtype)

        padded = torch.cat([pad, noised], dim=1)  # (B, pad_len+T, S, d)

        # idx[j,s] = pad_len + j - s  (select token j-s at level s)
        j = torch.arange(T, device=device).view(T, 1)
        s = torch.arange(self.n_step, device=device).view(1, self.n_step)
        idx = (pad_len + j - s).to(torch.long)  # (T,S)

        b_idx = torch.arange(B, device=device).view(B, 1, 1)
        x = padded[b_idx, idx.view(1, T, self.n_step), s.view(1, 1, self.n_step), :]  # (B,T,S,d)

        # mask: True where (j - s) >= 0 (i.e., corresponds to real token, not X)
        real = (j - s) >= 0  # (T,S), bool

        x_flat = x.reshape(B, T, self.n_embd)  # concat in s-major order (s then d)
        return x_flat, x, real

    def _latent_mse(
        self,
        pred: Tensor,  # (B, T-1, S, d)
        target: Tensor,  # (B, T-1, S, d)
        real_mask: Tensor,  # (T-1, S)
    ) -> Tensor:
        # Expand mask to (B, T-1, S, d)
        mask = real_mask.to(dtype=pred.dtype, device=pred.device).view(1, -1, self.n_step, 1)
        se = (pred - target) ** 2
        num = (se * mask).sum()
        den = mask.sum().clamp_min(1.0)
        return num / den

    def forward(self, toks: Tensor, state: tuple[Any, Any], _targets = None):
        device = toks.device
        B, T = toks.shape
        diffusion_state, backbone_state = state

        if self.mode == "train":
            assert T > 1, f"Cannot train on single length sequences"
                
            # Teacher-forced packed inputs x_j.
            x_flat, x, real = self._prep_backbone_inputs(toks)

            # Add positional embeddings and run backbone.
            pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)
            x_in = self.drop(x_flat + self.wpe(pos))  # (B,T,n_embd)
            y_flat, new_backbone_state = self.backbone(x_in, backbone_state)  # (B,T,n_embd)
            y = y_flat.view(B, T, self.n_step, self.n_embd_per_step)  # y_j ≈ x_{j+1}

            # Token logits from cleanest sublatent of y_j, which corresponds to token j+1.
            tok_logits = self.lm_head(y[:, :, 0, :])  # (B,T,V)

            # Token loss: standard next-token CE (ignore final position).
            ce = F.cross_entropy(
                tok_logits[:, :-1, :].reshape(B * (T - 1), -1),
                toks[:, 1:].reshape(B * (T - 1)),
            )
            # Latent loss: y[:, :-1] vs x[:, 1:]
            latent = self._latent_mse(
                pred=y[:, :-1, :, :],
                target=x[:, 1:, :, :],
                real_mask=real[1:, :],
            )
            loss = ce + self.latent_loss_scale*latent

            # For training, we don't need to persist diffusion_state; return a lightweight value.
            new_diff_state = {
                "x": torch.randn(B, self.n_step, self.n_embd_per_step, device=device),
                "pos": int(T),
            }
            return tok_logits, (new_diff_state, new_backbone_state), loss

        # -------------------------
        # Sampling / streaming mode
        # -------------------------
        start = diffusion_state["pos"]
        # We'll return logits for all positions, but only fill the newly processed ones.
        tok_logits = torch.zeros(B, T, self.lm_head.out_features, device=device)

        cur_x = diffusion_state["x"]
        cur_pos = diffusion_state["pos"]
        new_backbone_state = backbone_state

        # Process only unseen tokens.
        for i in range(start, T):
            # Overwrite cleanest slot (s=0) with embedding of the newly seen token.
            cur_x = cur_x.clone()
            cur_x[:, 0, :] = self.wte(toks[:, i])

            # Backbone expects (B, 1, n_embd)
            x_step = cur_x.reshape(B, 1, self.n_embd)

            # Position embedding uses the token index within the current block.
            # (Caller is responsible for cropping toks to <= n_block in generate().)
            pos_idx = torch.tensor([i], device=device, dtype=torch.long)
            x_step = self.drop(x_step + self.wpe(pos_idx).view(1, 1, self.n_embd))

            y_step, new_backbone_state = self.backbone(x_step, new_backbone_state)  # (B,1,n_embd)
            y_step_u = y_step.view(B, self.n_step, self.n_embd_per_step)  # (B,S,d)

            # Emit logits predicting token i+1, from cleanest slot of predicted next latent.
            tok_logits[:, i, :] = self.lm_head(y_step_u[:, 0, :])

            # Advance the diffusion window.
            cur_x = y_step_u
            cur_pos = i + 1

        diffusion_state = cast(_DiffusionState, {"x": cur_x.detach(), "pos": cur_pos})
        return tok_logits, (diffusion_state, new_backbone_state), None

    @torch.no_grad()
    def generate(self, tok: Tensor, max_new_tokens: int, state, temperature: float = 1.0, top_k: Optional[int] = None):
        """Autoregressive token generation using the model in sample mode."""
        for _ in range(max_new_tokens):
            # crop to block size
            tok = tok if tok.size(1) <= self.n_block else tok[:, -self.n_block:]

            logits, state, _ = self(tok, state)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            tok = torch.cat((tok, x_next), dim=1)
        return tok

    def initial_state(self, batch_size: int):
        # diffusion_state is a dict in sample mode; in train mode it's ignored.
        diffusion_state: _DiffusionState = {
            "x": torch.empty(batch_size, self.n_step, self.n_embd_per_step, device=self.device),
            "pos": 0,
        }
        # allocate proper device later in forward if empty
        return (diffusion_state, self.backbone.initial_state(batch_size))

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params
