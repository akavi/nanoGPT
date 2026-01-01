from dataclasses import dataclass
from typing import Literal, Optional, Any
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

class ArDiffusion(nn.Module):
    def __init__(self, config, backbone: nn.Module):
        super().__init__()
        assert config.n_vocab is not None
        assert config.n_block is not None
        self.mode = config.mode
        self.n_block = config.n_block
        self.n_step = config.n_step
        self.latent_loss_scale = config.latent_loss_scale
        self.n_embd = config.n_embd
        self.device = config.device

        self.n_embd_per_step = config.n_embd // config.n_step
        self.wte = nn.Embedding(config.n_vocab, self.n_embd_per_step)
        self.wpe = nn.Embedding(config.n_block + config.n_step - 1, self.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=False)

        self.lm_head = nn.Linear(self.n_embd_per_step, config.n_vocab, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

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
    ) -> tuple[Tensor, Tensor]:
        device = toks.device
        b, t = toks.size()
        assert t <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"


        # construct mask
        side_negative_mask = torch.zeros(1, self.n_step - 1, self.n_step, 1, device=device)
        main_positive_mask = torch.ones(1, t, self.n_step, 1, device=device)
        mask = tilt(
            torch.concat([side_negative_mask, main_positive_mask, side_negative_mask], dim=1),
            tilt_dim=2,
            content_dim=1,
        ) # (1, t + n_step - 1, n_step, 1)

        emb_toks = self.wte(toks) # token embeddings of shape (b, t, n_embd)
        assert emb_toks.shape[-1] == self.n_embd_per_step, (emb_toks.shape, self.n_embd_per_step)
        exp_emb_toks= emb_toks.unsqueeze(-2).expand(
            *emb_toks.shape[:-1], 
            self.n_step, 
            emb_toks.shape[-1]
        ) # (b, t, n_step, n_embd_per_step)


        left_noise  = torch.randn(b, self.n_step - 1, self.n_step, self.n_embd_per_step, device=device)
        right_noise = torch.randn(b, self.n_step - 1, self.n_step, self.n_embd_per_step, device=device)
        noise = torch.randn(b, t, 1, self.n_embd_per_step, device=device)   # (B,T,1,E)
        # weights: include 0.0 (clean), exclude 1.0 (pure noise)
        w = torch.linspace(0.0, 1.0, steps=self.n_step + 1, device=device)[: self.n_step]
        w = w.view(1, 1, self.n_step, 1)                                  # (1,1,S,E)
        noi_exp_emb_toks = exp_emb_toks * (1.0 - w) + noise * w

        cat_noi_exp_emb_toks = torch.concat([left_noise, noi_exp_emb_toks, right_noise], dim=1)
        # Tilt along step dimension, truncate along sequence dimension
        x_in = tilt(cat_noi_exp_emb_toks, tilt_dim=2, content_dim=1) # (b, t + n_step - 1, n_step, n_embd_per_step)
        return x_in, mask

    # For training
    # tok:   [tok1, tok2, tok3, tok4, tok5]
    # embed(tok):   [emb1, emb2, emb3, emb4, emb5] # each token embedded in n_embd/num_steps 
    # Note, read D(n, m) as "the mth diffusion step of the embedding of the nth token"
    # diffusion [
    #    [D(1, 1), D(1, 2), D(1, 3), ...D(1, n_steps)]
    #    [D(2, 1), D(2, 2), D(2, 3), ...D(2, n_steps)]
    #    [D(3, 1), D(3, 2), D(3, 3), ...D(3, n_steps)]
    #    [D(4, 1), D(4, 2), D(4, 3), ...D(4, n_steps)]
    #    [D(5, 1), D(5, 2), D(5, 3), ...D(5, n_steps)]
    # ] // (B, T, n_embd/num_steps)
    # reshaped [
    #    [0, 0, 0, ..., D(1, 1)]
    #    [0, 0, ..., D(1, 2), D(2, 1)]
    #    [0, ..., D(1, 3), D(2, 2), D(3, 1)]
    #    ...
    #    [D(1, n_steps), D(2, n_steps -1), D(3, n_steps -2), ..., D(n_steps, 1)]
    #    ...and so on
    # ] // (B, T, n_embd/num_steps)
    #  if mode == train, loss is crossentropy wrt to next position in "diffusion" array
    # if mode == sample, we need to do prefill for the "triangle" getting the newest tokens and then store it in the state
    # (updating the state for each subsequent sequence position

    def forward(self, toks, state):
        device = toks.device
        diffusion_state, backbone_state = state
        x_in, mask = self._prep_backbone_inputs(toks)
        (B, T), L = toks.size(), x_in.shape[1]

        if self.mode == "train":
            y, new_backbone_state = self._one_step(x_in, backbone_state)
            tok_logits = self.lm_head(y[:, self.n_step:, -1, :])

            ce = F.cross_entropy(
                tok_logits.reshape(B * (T - 1), -1),
                toks[:, 1:].reshape(B * (T - 1)),
            )

            # Latent loss: y[:, :-1] vs x[:, 1:]
            latent = _latent_mse(
                pred=y[:, :-1, :, :],          # (B, L-1, S, E)
                target=x_in[:, 1:, :, :],      # (B, L-1, S, E)
                real_mask=mask[:, 1:, :, :],   # (1, L-1, S, 1) broadcast over B/E
            )
            loss = ce + self.latent_loss_scale*latent

            # For training, we don't need to persist diffusion_state; return a lightweight value.
            new_diff_state = y
            return tok_logits, (new_diff_state, new_backbone_state), loss
        else: # self.mode == "sample"
            prefill_length = toks.shape[1] + self.n_step - 1
            for idx in range(diffusion_state.shape[1], prefill_length):
                diffusion_state, new_backbone_state = self._one_step(x_in[:, idx:, :, :], backbone_state)
                # update using mask, x_in, diffusion_state
                x_in = x_in # TODO

            y, new_backbone_state = self._one_step(x_in, backbone_state)
            # Token logits from cleanest sublatent of y_j, which corresponds to token j+1.
            tok_logits = self.lm_head(y[:, :, -1, :])  # (B,T,V)
            diffusion_state = y
            return tok_logits, (diffusion_state, new_backbone_state), None

    @torch.no_grad()
    def generate(self, tok, max_new_tokens, state, temperature=1.0, top_k=None):
        # implementation from 
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
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
        diffusion_state = torch.empty(batch_size, 0, 0, 0, device=self.device)
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

    def _one_step(self, x: Tensor, backbone_state):
        B, L, _, _ = x.shape()
        x_flat = x.reshape(B, L, self.n_step * self.n_embd_per_step)
        pos = torch.arange(0, L, dtype=torch.long, device=self.device) # shape (t)
        pos_emb = self.wpe(pos) # position embeddings of shape (L, n_embd)
        x_flat = self.drop(pos_emb + x_flat)

        # forward the GPT model itself
        y_flat, new_backbone_state = self.backbone(x_flat, backbone_state)  # (B,L,n_embd)
        y_flat = self.ln_f(y_flat)
        y = y_flat.view(B, L, self.n_step, self.n_embd_per_step)                        # (B, L, S, E)
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
    idx = (rows + offs)                                       # [T,K]

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
