from dataclasses import dataclass
from typing import Literal, Optional, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor, Tensor, nn
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

class ArDiffusion(nn.Module):
    def __init__(self, config, backbone: nn.Module):
        super().__init__()
        assert config.n_vocab is not None
        assert config.n_block is not None
        self.mode = config.mode
        self.n_block = config.n_block
        self.n_step = config.n_step

        self.n_embd_per_step = config.n_embd // config.n_step
        self.wte = nn.Embedding(config.n_vocab, self.n_embd_per_step)
        self.wpe = nn.Embedding(config.n_block, self.n_embd)
        self.backbone = backbone
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd, bias=False)

        self.lm_head = nn.Linear(self.n_embd_per_step, config.n_vocab, bias=False)
        # Weight tying: share embeddings between input and output layers
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
        b, t = toks.size()
        diffusion_state, backbone_state = state
        if self.mode == "train":
            assert t <= self.n_block, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"
            assert t >= self.n_step, f"Cannot forward sequence of length {t}, block size is only {self.n_block}"

            emb_toks = self.wte(toks) # token embeddings of shape (b, t, n_embd)
            exp_emb_toks= emb_toks.unsqueeze(-2).expand(
                *emb_toks.shape[:-1], 
                self.n_step, 
                emb_toks.shape[-1]
            ) # (b, t, n_step, n_embd_per_step)

            w = torch.linspace(0.0, 1.0, steps=self.n_step + 1, device=device).view(1, 1, self.n_step + 1, 1)[:, :, 1:, :]
            noise = torch.randn(b, t, self.n_step, self.n_embd_per_step, device=device)
            noi_exp_emb_toks = exp_emb_toks * (1.0 - w) + noise * w   
            left_noise = torch.randn(b, self.n_step, self.n_step, self.n_embd_per_step, device=device)
            right_noise = torch.randn(b, self.n_step, self.n_step, self.n_embd_per_step, device=device)
            cat_noi_exp_emb_toks = torch.concat([left_noise, noi_exp_emb_toks, right_noise], dim=1)
            tilt_noi_exp_emb_toks = tilt(cat_noi_exp_emb_toks, tilt_dim=-2, content_dim=-3)
            # Flatten sub-latents: (b, seq, n_step, n_embd_per_step) -> (b, seq, n_embd)
            b_dim, seq_dim = tilt_noi_exp_emb_toks.shape[:2]
            cat_tilt_noi_exp_emb_toks = tilt_noi_exp_emb_toks.reshape(b_dim, seq_dim, self.n_embd_per_step * self.n_step)

            # forward the GPT model itself
            pos = torch.arange(0, seq_dim, dtype=torch.long, device=device) # shape (seq_dim)
            emb_pos = self.wpe(pos) # position embeddings of shape (seq_dim, n_embd)
            x = self.drop(cat_tilt_noi_exp_emb_toks + emb_pos)
            new_x, new_backbone_state = self.backbone(x, backbone_state)

            # Slice the topmost sub-latent (first n_embd_per_step dims) to generate logits
            topmost_latent = new_x[:, :, :self.n_embd_per_step]  # (b, seq_dim, n_embd_per_step)
            tok_logits = self.lm_head(topmost_latent)  # (b, seq_dim, n_vocab)

            # Calculate losses - shift to predict next position
            # Logit loss: cross-entropy against the next token in the diagonal sequence
            # We need to map back to which actual tokens these correspond to
            # For now, compute loss on positions that have valid next targets
            logit_loss = F.cross_entropy(
                tok_logits[:, :-1, :].reshape(-1, tok_logits.size(-1)),
                toks.reshape(-1),
                reduction='mean'
            ) if t > 0 else torch.tensor(0.0, device=device)

            # Embedding loss: MSE between current embeddings and next position embeddings
            emb_loss = F.mse_loss(new_x[:, :-1, :], new_x[:, 1:, :].detach(), reduction='mean')

            loss = logit_loss + emb_loss

            # Store the last n_step positions of the sequence as diffusion state
            new_diffusion_state = new_x[:, -self.n_step:, :] if seq_dim >= self.n_step else new_x
            return tok_logits, (new_diffusion_state, new_backbone_state), loss
        else: # self.mode == "sample"
            # Check if we need to prefill (state should be dim 1 for sequence length)
            if diffusion_state.shape[1] < self.n_step:
                # Prefill: build up the initial diffusion state
                # a) Embed the passed token
                assert toks.size(1) == 1, "Prefill expects a single token"
                emb_tok = self.wte(toks.squeeze(1))  # (b, n_embd_per_step)

                # b) Create noise progression as in training
                w = torch.linspace(0.0, 1.0, steps=self.n_step + 1, device=device).view(1, self.n_step + 1, 1)[:, 1:, :]
                noise = torch.randn(b, self.n_step, self.n_embd_per_step, device=device)
                noi_emb_tok = emb_tok.unsqueeze(1) * (1.0 - w) + noise * w  # (b, n_step, n_embd_per_step)

                # c) Initialize diffusion state - will be built up iteratively
                # Start from the noisiest step and denoise progressively
                current_state = torch.zeros(b, 0, self.n_embd, device=device)
                current_backbone_state = backbone_state

                # d) Iteratively denoise from step n_step down to step 1
                for step_idx in range(self.n_step):
                    # Build the current latent vector by concatenating sub-latents
                    # The topmost sub-latent is from the current denoising step
                    topmost = noi_emb_tok[:, step_idx, :]  # (b, n_embd_per_step)

                    # Pad with zeros for the remaining sub-latents (or use previous state)
                    if step_idx == 0:
                        # First step: only topmost sub-latent, rest are zeros
                        current_latent = torch.cat([
                            topmost,
                            torch.zeros(b, self.n_embd - self.n_embd_per_step, device=device)
                        ], dim=-1).unsqueeze(1)  # (b, 1, n_embd)
                    else:
                        # Subsequent steps: use previous sub-latents from state
                        prev_sublatents = current_state[:, -1, self.n_embd_per_step:]  # (b, remaining_embd)
                        padding_needed = self.n_embd - self.n_embd_per_step - prev_sublatents.size(-1)
                        if padding_needed > 0:
                            prev_sublatents = torch.cat([
                                prev_sublatents,
                                torch.zeros(b, padding_needed, device=device)
                            ], dim=-1)
                        else:
                            prev_sublatents = prev_sublatents[:, :self.n_embd - self.n_embd_per_step]
                        current_latent = torch.cat([topmost, prev_sublatents], dim=-1).unsqueeze(1)  # (b, 1, n_embd)

                    # Add positional embedding
                    pos = torch.tensor([step_idx], dtype=torch.long, device=device)
                    emb_pos = self.wpe(pos)  # (1, n_embd)
                    x = self.drop(current_latent + emb_pos)

                    # Pass through backbone
                    new_x, current_backbone_state = self.backbone(x, current_backbone_state)

                    # Append to state
                    current_state = torch.cat([current_state, new_x], dim=1)

                # Update the diffusion state to the last n_step positions
                new_diffusion_state = current_state[:, -self.n_step:, :]
                new_backbone_state = current_backbone_state

                # Extract topmost sub-latent for logits
                topmost_latent = new_diffusion_state[:, -1, :self.n_embd_per_step]  # (b, n_embd_per_step)
                tok_logits = self.lm_head(topmost_latent).unsqueeze(1)  # (b, 1, n_vocab)

                return tok_logits, (new_diffusion_state, new_backbone_state), None

            else:
                # Normal forward pass after prefill
                # Embed the newest token as the topmost sub-latent
                assert toks.size(1) == 1, "Sample mode expects a single token"
                emb_tok = self.wte(toks.squeeze(1))  # (b, n_embd_per_step)

                # Use previous sub-latents from diffusion state
                prev_sublatents = diffusion_state[:, -1, self.n_embd_per_step:]  # (b, remaining_embd)
                current_latent = torch.cat([emb_tok, prev_sublatents], dim=-1).unsqueeze(1)  # (b, 1, n_embd)

                # Add positional embedding
                seq_pos = diffusion_state.shape[1]
                pos = torch.tensor([seq_pos], dtype=torch.long, device=device)
                emb_pos = self.wpe(pos)  # (1, n_embd)
                x = self.drop(current_latent + emb_pos)

                # Pass through backbone
                new_x, new_backbone_state = self.backbone(x, backbone_state)

                # Update diffusion state (shift and append)
                new_diffusion_state = torch.cat([diffusion_state[:, 1:, :], new_x], dim=1)

                # Take the topmost sub-latent and output logits
                topmost_latent = new_x[:, -1, :self.n_embd_per_step]  # (b, n_embd_per_step)
                tok_logits = self.lm_head(topmost_latent).unsqueeze(1)  # (b, 1, n_vocab)

                return tok_logits, (new_diffusion_state, new_backbone_state), None

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
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            tok = torch.cat((tok, x_next), dim=1)

        return tok
    
    
    def initial_state(self, batch_size):
        diffusion_state = torch.empty(batch_size, 0, 0, 0)
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
