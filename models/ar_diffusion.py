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
            # TODO: is this right? 
            assert t >= self.n_step, f"Cannot train on sequence of length {t}, n_step is {self.n_step}"

            emb_toks = self.wte(toks) # token embeddings of shape (b, t, n_embd)
            exp_emb_toks= emb_toks.unsqueeze(-2).expand(
                *emb_toks.shape[:-1], 
                self.n_step, 
                emb_toks.shape[-1]
            ) # (b, t, n_step, n_embd_per_step)

            w = torch.linspace(0.0, 1.0, steps=self.n_step + 1, device=device).view(1, 1, self.n_step + 1, 1)[:, :, 1:, :]
            noise = torch.randn(b, t, self.n_step, self.n_embd_per_step, device=device)
            noi_exp_emb_toks = exp_emb_toks * (1.0 - w) + noise * w   
            left_noise = torch.randn(b, self.n_step - 1, self.n_step, self.n_embd_per_step)
            right_noise = torch.randn(b, self.n_step - 1, self.n_step, self.n_embd_per_step)
            # Concat along sequence dimension
            cat_noi_exp_emb_toks = torch.concat([left_noise, noi_exp_emb_toks, right_noise], dim=1)
            # Tilt along step dimension, truncate along sequence dimension
            tilt_noi_exp_emb_toks = tilt(cat_noi_exp_emb_toks, tilt_dim=2, content_dim=1) # (b, t, n_step, n_embd_per_step)
            cat_tilt_noi_exp_emb_toks = tilt_noi_exp_emb_toks.reshape(b, t, self.n_embd) # (b, t, n_embd)

            # forward the GPT model itself
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            emb_pos = self.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.drop(cat_tilt_noi_exp_emb_toks + emb_pos)
            new_x, new_backbone_state = self.backbone(x, backbone_state)
            exp_new_x = new_x.unsqueeze(-2).expand(
                *new_x.shape[:-1], 
                self.n_step, 
                emb_toks.shape[-1]
            ) # (b, t, n_step, n_embd_per_step)

            # TODO:
            # 4) slice the "topmost" sub_latent and use it to generate tok_logits
            # 5) calculate logit loss (KL of tok_logits against tok at previous sequence index)
            # 6) calculate embedding loss (KL of new_x against x at previous sequence index, masking out the right noise)

            # TODO: 7) Is this right?
            new_diffusion_state = exp_new_x
            # TODO:
            return tok_logits, (new_diffusion_state, new_backbone_state), loss
        else: # self.mode == "sample"
            state_length = diffusion_state.shape[1]
            idxs_needing_prefill = max(state_length - (toks.shape[1] + self.n_step - 1), 0)
            idxs_needing_generation = max(state_length -  (toks.shape[1] + self.n_step - 1), 0) # This is wrong, TODO
            if diffusion_state.shape[1] < toks.shape[1] + self.n_step - 1
                # TODO 8) perform prefill of diffusion_state by 
                # a) embedding the passed token
                # b) creating a noise progression as before
                # c) diagonalizing it
                # d) then passing the latents through the backbone,
                #    starting with the noisiest. After each pass, take the output
                #    latent, and _replace_  the output sublatents corresponding 
                #    to positions generated by the diagnoalized noise progression 
                #    with said generated sublatents
            # TODO 9) once we have a prefill then we do a "normal" forward pass 
            # to the backbone by embedding the newest passed token as the "topmost" sub_latent
            # TODO 10) take the "topmost" sub_latent and use it to output logits

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
