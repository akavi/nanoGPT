"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, asdict
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import LongTensor, Tensor, nn
from einops import rearrange, repeat
from models.layer_norm import LayerNorm

class ModuleList(nn.Module):
    def __init__(self, inner_list):
        super().__init__()
        self.inner_list = nn.ModuleList(inner_list)

    def forward(self, x, old_state):
        new_state = []

        for idx, fn in enumerate(self.inner_list):
            x, this_state = fn(x, old_state[idx])
            new_state.append(this_state)

        return x, new_state

    def initial_state(self, batch_size):
        return list(map(lambda m: m.initial_state(batch_size), self.inner_list))

    def flops_per_fwdbwd(self):
        return sum(list(map(lambda m: m.flops_per_fwdbwd(), self.inner_list)))

# If you can't be assed to implement initial_state/flops_per_fwdbwd correctly)
class LazyWrapper(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def forward(self, x, old_state):
        return self.impl(x), old_state

    def initial_state(self, batch_size):
        return None

    def flops_per_fwdbwd(self):
        return 0


def _rope_freqs(head_dim: int, max_len: int, device: torch.device, base: float = 10000.0) -> Tensor:
    """Precompute complex RoPE frequencies: (max_len, head_dim//2) complex64."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_len, device=device).float()
    angles = torch.outer(t, freqs)  # (max_len, head_dim//2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def _apply_rope(x: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary embeddings to x: (B, n_head, T, head_dim)."""
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rot = x_complex * freqs
    return torch.view_as_real(x_rot).flatten(-2).to(x.dtype)


def apply_rope_1d(q: Tensor, k: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Standard 1D RoPE. q, k: (B, n_head, T, head_dim). positions: (T,) int."""
    head_dim = q.shape[3]
    max_pos = int(positions.max().item()) + 1
    freqs = _rope_freqs(head_dim, max_pos, q.device)
    freqs = freqs[positions].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim//2)
    return _apply_rope(q, freqs), _apply_rope(k, freqs)


def apply_rope_2d(q: Tensor, k: Tensor, positions: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """2D RoPE using (x,y) coordinate lookup. coords: (block_size, 2) int tensor.
    positions: (T,) int indices into coords. Splits head_dim in half for x and y."""
    head_dim = q.shape[3]
    half = head_dim // 2
    max_coord = int(coords.max().item()) + 1
    freqs_x = _rope_freqs(half, max_coord, q.device)
    freqs_y = _rope_freqs(half, max_coord, q.device)

    pos_coords = coords[positions]  # (T, 2)
    fx = freqs_x[pos_coords[:, 0]].unsqueeze(0).unsqueeze(0)  # (1, 1, T, half//2)
    fy = freqs_y[pos_coords[:, 1]].unsqueeze(0).unsqueeze(0)
    freqs = torch.cat([fx, fy], dim=-1)  # (1, 1, T, head_dim//2)

    return _apply_rope(q, freqs), _apply_rope(k, freqs)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx, rope_fn=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.layer_idx = layer_idx
        self.attn_sums = {}
        self.attn_counts = {}
        self.use_decay = config.use_decay
        self.dropout = config.dropout
        self.rope_fn = rope_fn

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.decay_raw = nn.Parameter(torch.zeros(self.n_head)) if self.use_decay else None

    def forward(self, x, state, positions=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        hs = C // self.n_head

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)

        # fold: concatenate with cached k,v (identity = zero-length tensors)
        cached_k, cached_v = state
        offset = cached_k.size(2)

        # Apply RoPE before concatenating with cache (cache already has RoPE applied)
        if self.rope_fn is not None:
            if positions is None:
                positions = torch.arange(offset, offset + T, device=x.device)
            q, k = self.rope_fn(q, k, positions)
        k = torch.cat([cached_k, k], dim=2)
        v = torch.cat([cached_v, v], dim=2)
        T_full = k.size(2)

        # causal mask: new queries attend to all cached keys + causal within new tokens
        neg_inf = torch.finfo(torch.float32).min
        attn_mask = torch.triu(torch.full((T, T), neg_inf, device=x.device, dtype=torch.float32), diagonal=1)
        if offset > 0:
            attn_mask = torch.cat([
                torch.zeros(T, offset, device=x.device, dtype=torch.float32),
                attn_mask,
            ], dim=1)

        if self.use_decay:
            decay = F.softplus(self.decay_raw).to(q.dtype)                 # (nh,)
            q_idx = torch.arange(offset, offset + T, device=x.device)
            k_idx = torch.arange(T_full, device=x.device)
            dist = (q_idx[:, None] - k_idx[None, :]).clamp_min(0).to(torch.float32)
            log1p_dist = torch.log1p(dist)                         # (T, T_full)
            logw = -torch.log1p(decay.view(self.n_head, 1, 1) * log1p_dist)
            attn_mask = attn_mask + logw

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, (k, v)

class MLP(nn.Module):

    def __init__(self, n_embd, bias = False, dropout = 0.0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

@dataclass
class CsaConfig:
    n_head: int
    n_embd: int
    n_step: int
    block_size: int
    bias: bool
    dropout: float
    use_decay: bool = False

class CsaBlock(nn.Module):

    def __init__(self, config, layer, rope_fn=None):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer, rope_fn=rope_fn)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config.n_embd, bias=config.bias, dropout=config.dropout)

    def forward(self, x, state, positions=None):
        attn_out, state = self.attn(self.ln_1(x), state, positions=positions)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, state

    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        hs = self.n_embd // self.n_head
        empty = torch.zeros(batch_size, self.n_head, 0, hs, device=device)
        return (empty, empty)

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def flops_per_fwdbwd(self):
        N = self.get_num_params()
        H, Q, T = self.n_head, self.n_embd // self.n_head, self.block_size
        flops_per_token = 6*N + 12*H*Q*T
        return flops_per_token * T


class SubLatentCsaBlock(nn.Module):
    def __init__(self, config: CsaConfig, layer: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_step = config.n_step
        self.block_size = config.block_size

        assert self.n_embd % self.n_step == 0, (self.n_embd, self.n_step)
        self.n_embd_per_step = self.n_embd // self.n_step

        # LN is now per-latent: normalize over E, not over (S*E)
        self.ln_1 = LayerNorm(self.n_embd_per_step, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer)
        self.ln_2 = LayerNorm(self.n_embd_per_step, bias=config.bias)
        self.mlp = MLP(config.n_embd, bias=config.bias, dropout=config.dropout)

    def _per_latent_ln(self, x: torch.Tensor, ln: nn.Module) -> torch.Tensor:
        """
        x: (B, T, S*E) -> reshape to (B, T, S, E), LN over E, flatten back.
        """
        B, T, D = x.shape
        S, E = self.n_step, self.n_embd_per_step
        # view/reshape is fine as long as x is contiguous; reshape handles non-contig safely.
        x4 = x.reshape(B, T, S, E)
        x4 = ln(x4)                    # LN over last dim (E)
        return x4.reshape(B, T, S * E)

    def forward(self, x, state):
        attn_out, state = self.attn(self._per_latent_ln(x, self.ln_1), state)
        x = x + attn_out
        x = x + self.mlp(self._per_latent_ln(x, self.ln_2))
        return x, state

    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        hs = self.n_embd // self.n_head
        empty = torch.zeros(batch_size, self.n_head, 0, hs, device=device)
        return (empty, empty)

    def get_num_params(self, non_embedding=True):
        return sum(p.numel() for p in self.parameters())

    def flops_per_fwdbwd(self):
        N = self.get_num_params()
        H, Q, T = self.n_head, self.n_embd // self.n_head, self.block_size
        flops_per_token = 6 * N + 12 * H * Q * T
        return flops_per_token * T

@dataclass
class GPTConfig:
    n_state: int = 128
    mode: str = "train"
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_inner: int = 1536 # n_embd * 2 ?
    n_conv: int = 4
    n_chunk: int = 64
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_decay: bool = False
    block_type: Literal["attention", "mamba"] = "attention"
    device: str = "cuda"

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.block_type == "attention":
            backbone = nn.ModuleList([CsaBlock(config, i) for i in range(config.n_layer)])
        elif config.block_type == "mamba":
            backbone = nn.ModuleList([
                Mamba2(LayerConfig(
                    mode=config.mode,
                    block_size=config.block_size,
                    n_head=config.n_head,
                    n_embd=config.n_embd,
                    n_inner=config.n_inner,
                    n_conv=config.n_conv,
                    n_state=config.n_state,
                    n_chunk=config.n_state,
                    dropout=config.dropout,
                    bias=config.bias,
                    use_decay=config.use_decay,
                    device=config.device,
                ), i) for i in range(config.n_layer)
            ])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            backbone = backbone,
            drop = nn.Dropout(config.dropout),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def flops_per_fwdbwd(self): 
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        return flops_per_token * T

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, state, targets=None):
        device = idx.device
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # only process tokens not already in the KV-cache
        offset = state[0][0].size(2)  # cached seq len
        idx_new = idx[:, offset:]
        t_new = idx_new.size(1)
        pos = torch.arange(offset, offset + t_new, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx_new)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.backbone):
            x, state[i] = block(x, state[i])
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, state, loss, {}


    def optim_groups(self) -> list[dict]:
        from utils import decay_nodecay_groups
        return decay_nodecay_groups(self.parameters())

    def initial_state(self, batch_size):
        return list(map(lambda block: block.initial_state(batch_size), self.transformer.backbone))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            logits, state, _, _ = self(idx, state)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
