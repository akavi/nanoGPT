from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor, nn

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
        self.n_embd = config.n_embd
        self.device = config.device

        self.n_embd_per_step = config.n_embd // config.n_step
        self.wte = nn.Embedding(config.n_vocab, self.n_embd_per_step)
        self.wpe = nn.Embedding(config.n_block + config.n_step - 1, self.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.in_norm = SubLatentLayerNorm(self.n_step, self.n_embd_per_step)
        self.out_norm = SubLatentLayerNorm(self.n_step, self.n_embd_per_step)

        self.lm_projection = nn.Linear(self.n_embd_per_step, config.n_vocab, bias=False)

        # init all weights
        self.apply(self._init_weights)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def _blend_fracs(self, device):
        """Per-sublatent blend fractions shaped (1, 1, S, 1) for broadcasting."""
        s_idx = torch.arange(self.n_step, device=device, dtype=torch.float)
        clean = (s_idx / self.n_step).view(1, 1, -1, 1)
        non_clean = ((self.n_step - s_idx) / self.n_step).view(1, 1, -1, 1)
        return clean, non_clean / self.n_step, non_clean * (self.n_step - 1) / self.n_step

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _prep_backbone_inputs(
        self,
        toks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Construct blended inputs with clean + AR + noise, then tilt.

        At pre-tilt position j, sublatent s (0-indexed):
            clean_frac = s / S
            ar_frac    = (S - s) / S²
            noise_frac = (S - 1)(S - s) / S²
            input = clean_frac * norm_emb(tok_j) + ar_frac * norm_emb(tok_{j-1}) + noise_frac * noise

        After tilting, position i, sublatent s corresponds to tok_{i-s},
        with AR from tok_{i-s-S} (the fully-denoised token from S steps back).

        When S=1: clean_frac=0, ar_frac=1, noise_frac=0 → vanilla AR (AR from tok_{i-1}).

        Returns:
          x_in:         (B, L, S, E) blended tilted input
          clean_tilted: (B, L, S, E) clean normed embeddings (MSE target)
          train_mask:   (1, L, S, 1)
          gen_mask:     (1, L, S, 1)
        """
        device = toks.device
        B, t = toks.size()
        assert t <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"

        S = self.n_step
        E = self.n_embd_per_step
        D = t + 2 * (S - 1)  # padded pre-tilt length

        side_neg = torch.zeros(1, S - 1, S, 1, device=device)
        main_pos = torch.ones(1, t, S, 1, device=device)
        train_mask = tilt(
            torch.cat([side_neg, main_pos, side_neg], dim=1),
            tilt_dim=2, content_dim=1,
        )  # (1, L, S, 1)

        # ---- Normalized token embeddings ----
        emb_normed = self.in_norm(self.wte(toks))  # (B, t, E)

        # Clean embeddings in pre-tilt space (valid at j = S-1 .. S-1+t-1)
        clean_pre = torch.zeros(B, D, E, device=device)
        clean_pre[:, S - 1 : S - 1 + t, :] = emb_normed

        # AR embeddings in pre-tilt space, shifted by 2S-1 so that after tilting,
        # AR at (i, s) = emb(toks[i - s - S]): each sublatent gets the token that
        # was fully denoised when that sublatent's token first entered the ladder.
        ar_pre = torch.zeros(B, D, E, device=device)
        ar_offset = 2 * S - 1
        n_ar = min(t, D - ar_offset)
        if n_ar > 0:
            ar_pre[:, ar_offset : ar_offset + n_ar, :] = emb_normed[:, :n_ar, :]

        # Expand to (B, D, S, E) — same embedding at every sublatent
        clean_exp = clean_pre.unsqueeze(2).expand(B, D, S, E)
        ar_exp = ar_pre.unsqueeze(2).expand(B, D, S, E)

        # Noise
        noise = torch.randn(B, D, S, E, device=device)

        # Blend and tilt
        cf, af, nf = self._blend_fracs(device)
        blended = cf * clean_exp + af * ar_exp + nf * noise
        x_in = tilt(blended, tilt_dim=2, content_dim=1)          # (B, L, S, E)
        clean_tilted = tilt(clean_exp, tilt_dim=2, content_dim=1) # (B, L, S, E)

        return x_in, clean_tilted, train_mask

    def forward(self, toks, state, targets=None):
        output_buf, backbone_state = state

        (B, T) = toks.size()
        S = self.n_step
        E = self.n_embd_per_step

        if self.mode == "train":
            x_in, clean_tilted, train_mask = self._prep_backbone_inputs(toks)
            L = x_in.shape[1]
            V = self.n_vocab

            y, new_backbone_state = self._one_step(x_in, backbone_state)
            tok_logits = self.lm_head(y)  # (B, L, S, V)
            new_diff_state = y

            # ---- Tilted targets (token indices) ----
            side_target = torch.zeros(B, S - 1, S, dtype=toks.dtype, device=toks.device)
            exp_toks = toks.unsqueeze(-1).expand(B, T, S)
            targets = tilt(
                torch.cat([side_target, exp_toks, side_target], dim=1),
                tilt_dim=2, content_dim=1,
            )  # (B, L, S)


            # CE loss on cleanest sublatent only (same position, no AR shift)
            ce_mask = train_mask[:, :, -1, 0]  # (1, L, S)
            ce_loss_per = F.cross_entropy(
                tok_logits[:, :, -1, :].reshape(B * L, V),
                targets[:, :, -1].reshape(B * L),
                reduction="none",
            ).reshape(B, L)
            ce_loss = (ce_loss_per * ce_mask).sum() / (B * ce_mask.sum()).clamp(min=1e-8)

            # MSE loss: all sublatents vs clean embeddings (x-prediction)
            latent_loss = _latent_mse(
                pred=y,
                target=clean_tilted.detach(),
                real_mask=train_mask,
            )

            loss = ce_loss + self.latent_loss_scale * latent_loss

            return tok_logits, (new_diff_state, new_backbone_state), loss

        else:  # sample — one position per call, AR from tok history
            cf, af, nf = self._blend_fracs(toks.device)
            idx = output_buf.shape[1] - 1  # position to process

            prev_out = output_buf[:, -1:, :, :]
            shifted = F.pad(prev_out[:, :, :-1, :], (0, 0, 1, 0))

            # AR signal: sublatent s wants emb(toks[idx - s - S]).
            # Pad with S zero-embeddings so out-of-range indices land on zeros.
            tok_emb = self.in_norm(self.wte(toks))  # (B, T, E)
            padded = F.pad(tok_emb, (0, 0, S, 0))  # (B, S + T, E)
            ar_indices = idx - torch.arange(S, device=toks.device)  # offset by S already from pad
            ar_emb = padded[:, ar_indices].unsqueeze(1)  # (B, 1, S, E)

            noise = torch.randn(B, 1, S, E, device=toks.device)
            blended = cf * shifted + af * ar_emb + nf * noise

            y, backbone_state = self._one_step(blended, backbone_state, pos_offset=idx)
            output_buf = torch.cat([output_buf, y[:, -1:, :, :]], dim=1)

            tok_logits = self.lm_head(y[:, -1:, -1, :])
            return tok_logits, (output_buf, backbone_state), None

    def lm_head(self, x: Tensor) -> Tensor:
        return self.lm_projection(x)  # (B,L,V)

    @torch.no_grad()
    def generate(self, tok, max_new_tokens, state, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.

        ARD does current-token prediction, so there's no seed/BOS. tok starts
        empty and accumulates the actually-selected tokens. The forward sample
        path reads AR embeddings directly from tok.

        For S > 1, the first S-1 calls warm up the sublatent ladder before
        the cleanest sublatent produces a valid prediction.
        """
        B = tok.shape[0]
        tok = torch.empty((B, 0), dtype=torch.long, device=self.device)

        # Warm up ladder: S-1 calls to fill the sublatent pipeline
        for _ in range(self.n_step - 1):
            _, state, _ = self(tok, state)

        for _ in range(max_new_tokens):
            logits, state, _ = self(tok, state)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            tok = torch.cat((tok, x_next), dim=1)
        return tok
    
    
    def initial_state(self, batch_size):
        output_buf = torch.zeros(batch_size, 1, self.n_step, self.n_embd_per_step, device=self.device)
        return (output_buf, self.backbone.initial_state(batch_size))

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()

    @torch.no_grad()
    def check_embedding_collisions(self, threshold=0.99):
        """Check if in_norm causes distinct tokens to have near-identical representations."""
        emb = self.wte.weight  # (V, E)
        normed = self.in_norm(emb.unsqueeze(0).unsqueeze(0)).squeeze()  # (V, E)
        # Normalize for cosine similarity
        normed = normed / (normed.norm(dim=-1, keepdim=True) + 1e-8)
        sims = torch.mm(normed, normed.t())  # (V, V)

        # Find near-duplicate pairs (excluding diagonal)
        mask = torch.triu(torch.ones_like(sims, dtype=torch.bool), diagonal=1)
        high_sim = (sims > threshold) & mask
        n_collisions = high_sim.sum().item()

        if n_collisions > 0:
            indices = high_sim.nonzero()
            print(f"Found {n_collisions} token pairs with cosine similarity > {threshold}:")
            for i, j in indices[:10]:  # show first 10
                print(f"  tokens {i.item()} and {j.item()}: sim={sims[i, j].item():.4f}")
            if n_collisions > 10:
                print(f"  ... and {n_collisions - 10} more")
        else:
            print(f"No token pairs with cosine similarity > {threshold}")

        return n_collisions, sims

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


    def _one_step(self, x: Tensor, backbone_state, pos_offset=0):
        B, L, _, _ = x.shape
        x_flat = x.reshape(B, L, self.n_step * self.n_embd_per_step)
        pos = torch.arange(pos_offset, pos_offset + L, device=self.device)
        pos_emb = self.wpe(pos)  # (L, n_embd)
        # Normalize pos_emb to zero mean, unit variance (matching in_norm on token embeddings)
        pos_emb = (pos_emb - pos_emb.mean(dim=-1, keepdim=True)) * torch.rsqrt(
            pos_emb.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
        )
        x_flat = self.drop(pos_emb + x_flat)

        y_flat, new_backbone_state = self.backbone(x_flat, backbone_state)  # (B,L,n_embd)

        y_pre = y_flat.view(B, L, self.n_step, self.n_embd_per_step)        # (B,L,S,E)
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
