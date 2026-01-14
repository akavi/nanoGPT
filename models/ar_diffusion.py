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
        self.n_embd = config.n_embd
        self.device = config.device

        self.n_embd_per_step = config.n_embd // config.n_step
        self.wte = nn.Embedding(config.n_vocab, self.n_embd_per_step)
        self.wpe = nn.Embedding(config.n_block + config.n_step - 1, self.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
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

    def _prep_backbone_inputs(
        self,
        toks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        device = toks.device
        b, t = toks.size()
        assert t <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"

        # construct mask
        side_negative_mask = torch.zeros(1, self.n_step - 1, self.n_step, 1, device=device)
        side_positive_mask = torch.ones(1, self.n_step - 1, self.n_step, 1, device=device)
        main_positive_mask = torch.ones(1, t, self.n_step, 1, device=device)
        train_mask = tilt(
            torch.concat([side_negative_mask, main_positive_mask, side_negative_mask], dim=1),
            tilt_dim=2,
            content_dim=1,
        ) # (1, t + n_step - 1, n_step, 1)
        gen_mask = tilt(
            torch.concat([side_positive_mask, main_positive_mask, side_negative_mask], dim=1),
            tilt_dim=2,
            content_dim=1,
        ) # (1, t + n_step - 1, n_step, 1)

        emb_toks = self.wte(toks) # token embeddings of shape (b, t, n_embd)
        # emb_toks = toks.unsqueeze(-1).repeat(1, 1, self.n_embd_per_step)# self.wte(toks) # token embeddings of shape (b, t, n_embd)
        assert emb_toks.shape[-1] == self.n_embd_per_step, (emb_toks.shape, self.n_embd_per_step)
        exp_emb_toks = emb_toks.unsqueeze(-2).expand(
            *emb_toks.shape[:-1], 
            self.n_step, 
            emb_toks.shape[-1]
        ) # (b, t, n_step, n_embd_per_step)

        noise = torch.randn(b, t, 1, self.n_embd_per_step, device=device)   # (B,T,1,E)

        # weight on clean: goes from 1/self.n_step to 1.0, excludes 0.0 (no clean)
        w = torch.linspace(0.0, 1.0, steps=self.n_step + 1, device=device)[1: ]

        w_base = torch.linspace(0.0, 1.0, steps=self.n_step + 1, device=device)[1:]  # [1/n, 2/n, ..., 1]
        w_base = w_base.view(1, 1, self.n_step, 1)                                  # (1,1,S,1)
        w = w_base ** (1 - self.gamma)
        noi_exp_emb_toks = w*exp_emb_toks + (1.0 - w)*noise

        left_pad  = torch.zeros(b, self.n_step - 1, self.n_step, self.n_embd_per_step, device=device)
        right_pad = torch.zeros(b, self.n_step - 1, self.n_step, self.n_embd_per_step, device=device)
        cat_noi_exp_emb_toks = torch.concat([left_pad, noi_exp_emb_toks, right_pad], dim=1)
        # Tilt along step dimension, truncate along sequence dimension
        x_in = tilt(cat_noi_exp_emb_toks, tilt_dim=2, content_dim=1) # (b, t + n_step - 1, n_step, n_embd_per_step)

        noise_exp = noise.expand(*exp_emb_toks.shape)
        cat_noise = torch.concat([
            torch.zeros(b, self.n_step - 1, self.n_step, self.n_embd_per_step, device=device),
            noise_exp,
            torch.zeros(b, self.n_step - 1, self.n_step, self.n_embd_per_step, device=device)
        ], dim=1)
        noise_tilted = tilt(cat_noise, tilt_dim=2, content_dim=1)
        
        return x_in, train_mask, gen_mask, noise_tilted

    # For training
    # tok:   [tok1, tok2, tok3, tok4, tok5]
    # embed(tok):   [emb1, emb2, emb3, emb4, emb5] # each token embedded in n_embd/num_steps 
    # Note, read D(n, m) as "the mth diffusion step of the embedding of the nth token"
    # diffusion [
    #    [D(1, 1), D(1, 2), D(1, 3), ...D(1, n_step)]
    #    [D(2, 1), D(2, 2), D(2, 3), ...D(2, n_step)]
    #    [D(3, 1), D(3, 2), D(3, 3), ...D(3, n_step)]
    #    [D(4, 1), D(4, 2), D(4, 3), ...D(4, n_step)]
    #    [D(5, 1), D(5, 2), D(5, 3), ...D(5, n_step)]
    # ] // (B, T, n_embd/num_steps)
    # reshaped [
    #    [0, 0, 0, ..., D(1, 1)]
    #    [0, 0, ..., D(1, 2), D(2, 1)]
    #    [0, ..., D(1, 3), D(2, 2), D(3, 1)]
    #    ...
    #    [D(1, n_step), D(2, n_step -1), D(3, n_step -2), ..., D(n_step, 1)]
    #    ...and so on
    # ] // (B, T, n_embd/num_steps)
    #  if mode == train, loss is crossentropy wrt to next position in "diffusion" array
    # if mode == sample, we need to do prefill for the "triangle" getting the newest tokens and then store it in the state
    # (updating the state for each subsequent sequence position
    def forward(self, toks, state, targets = None):
        diffusion_state, backbone_state = state
        # shaped_w intentionally
        x_in, train_mask, gen_mask, noise_tilted = self._prep_backbone_inputs(toks)
        # Because LLMs are incapable of using multi-letter variable names for
        # params
        (B, T), L, S, V = toks.size(), x_in.shape[1], self.n_step, self.n_vocab

        if self.mode == "train":
            y, new_backbone_state = self._one_step(x_in, backbone_state)
            tok_logits = self.lm_head(y)  # (B, L, S, V)
            new_diff_state = y

            with torch.no_grad():
                for s in range(S - 1):  # can't go beyond S-1
                    input_s = x_in[:, :-1, s, :]          # (B, L-1, E) - noisy
                    gt_cleaner = x_in[:, 1:, s+1, :]      # (B, L-1, E) - same token, cleaner
                    output_cleaner = y[:, :-1, s+1, :]    # (B, L-1, E) - model's attempt
                    
                    input_to_gt = ((input_s - gt_cleaner)**2).mean()
                    output_to_gt = ((output_cleaner - gt_cleaner)**2).mean()
                    
                    print(f"step {s}->{s+1}: input_to_gt={input_to_gt:.4f}, output_to_gt={output_to_gt:.4f}, ratio={output_to_gt/input_to_gt:.4f}")

            side_target_mask = torch.zeros(B, S - 1, S, dtype=toks.dtype, device=toks.device)  # (B, S-1, S)

            targets = toks.unsqueeze(-1).expand(B, T, S)  # (B, T, S)
            targets = tilt(
                torch.concat([side_target_mask, targets, side_target_mask], dim=1),
                tilt_dim=2,
                content_dim=1,
            ) # (1, t + n_step - 1, n_step)

            Ln = L - 1
            # logits for all slots
            ce_per = F.cross_entropy(
                tok_logits[:, :-1, :, :].reshape(B * Ln * S, V),
                targets[:, 1:, :].reshape(B * Ln * S),
                reduction="none",
            ).reshape(B, Ln, S)  # (B, Ln, S)

            # ---- mask: only count "real" ladder slots (and only where next-token exists) ----
            m = (train_mask[:, :-1, :, :] * train_mask[:, 1:, :, :]).squeeze(-1)          # (1, Ln, S)
            m_b = m.expand(B, Ln, S).to(ce_per.dtype)         # (B, Ln, S)

            ce_loss = (ce_per * m_b).sum() / m_b.sum().clamp_min(1.0)

            # MSE toward target (want to minimize)
            latent_loss = _latent_mse(
                pred=y[:, :-1, :, :],
                target=x_in[:, 1:, :, :].detach(),
                real_mask=train_mask[:, 1:, :, :],
            )
            loss = ce_loss + self.latent_loss_scale * latent_loss

            print(f"ce_loss={ce_loss.item()}, latent_loss={latent_loss.item()}")
            return tok_logits, (new_diff_state, new_backbone_state), loss

        else:  # self.mode == "sample"
            fill_length = toks.shape[1] + self.n_step - 1
            for idx in range(diffusion_state.shape[1] - 1, fill_length):
                m = gen_mask[:, idx:idx+1, :, :].to(dtype=x_in.dtype)  # (1,1,S,1) broadcast over B/E
                x_in[:, idx:idx+1, :, :] = m * x_in[:, idx:idx+1, :, :] + (1.0 - m) * diffusion_state[:, idx:idx+1, :, :]
                y, backbone_state = self._one_step(
                    x_in[:, :idx+1, :, :],
                    backbone_state,
                    pos_idx=idx,
                )
                diffusion_state = torch.concat([diffusion_state, y[:, -1:, :, :]], dim=1)

            tok_logits = self.lm_head(y[:, :, -1, :])  # (B,T,V) from cleanest sublatent
            return tok_logits, (diffusion_state, backbone_state), None


    def lm_head(self, x: Tensor) -> Tensor:
        x = self.pre_lm_head(x)  # (B,L,E)
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
    def __init__(self, S: int, E: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(S, E))  # gamma per S
            self.bias   = nn.Parameter(torch.zeros(S, E)) # beta per S
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, S, E)
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) * torch.rsqrt(var + self.eps)
        if self.elementwise_affine:
            y = y * self.weight[None, None, :, :] + self.bias[None, None, :, :]
        return y
