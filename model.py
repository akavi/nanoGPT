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
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import LongTensor, Tensor, nn
from einops import rearrange, repeat

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.layer_idx = layer_idx
        self.mode = config.mode
        self.attn_sums = {}
        self.attn_counts = {}
        self.use_decay = config.use_decay
        self.dropout = config.dropout

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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (Bil nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        neg_inf = torch.finfo(torch.float32).min
        attn_mask = torch.triu(torch.full((T, T), neg_inf, device=x.device, dtype=torch.float32), diagonal=1)
        if self.use_decay:
            decay = F.softplus(self.decay_raw).to(q.dtype)                 # (nh,)
            # dist[i,j] = max(i-j, 0)  
            idx  = torch.arange(T, device=x.device)
            dist = (idx[:, None] - idx[None, :]).clamp_min(0).to(torch.float32)  # (T,T)
            log1p_dist = torch.log1p(dist)                         # (T,T)
            logw = -torch.log1p(decay.view(self.n_head, 1, 1) * log1p_dist)   

            attn_mask += logw

        if self.mode == 'train':
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0, 
                is_causal=False
            )
        else: 
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attn_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            att_sum = att.detach().to(torch.float32).sum(dim=0)  # (nh, T, T), sum over batch
            if T not in self.attn_sums:
                self.attn_sums[T] = torch.zeros(
                    self.n_head, T, T, dtype=torch.float32, device=att_sum.device
                )
                self.attn_counts[T] = torch.zeros(
                    self.n_head, dtype=torch.float32, device=att_sum.device
                )
            self.attn_sums[T] += att_sum
            # each head gets +B examples
            self.attn_counts[T] += float(B)

            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CsaBlock(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, layer)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, state):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x, state

    def initial_state(self, batch_size):
        return None


InferenceCache = tuple[Tensor, Tensor]
class Mamba2(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.idx = idx
        self.device = config.device
        self.mode = config.mode

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.n_inner = config.n_inner
        self.headdim = self.n_inner // self.n_head
        self.n_state = config.n_state
        self.n_conv = config.n_conv
        self.n_chunk = config.n_chunk

        # Order: (z, x, B, C, dt)
        d_in_proj = 2 * self.n_inner + 2 * self.n_state + self.n_head
        self.in_proj = nn.Linear(self.n_embd, d_in_proj, bias=False, device=config.device)

        conv_dim = self.n_inner + 2 * self.n_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=self.n_conv,
            groups=conv_dim,
            padding=0,
            device=config.device,
        )

        self.dt_bias = nn.Parameter(torch.empty(self.n_head, device=config.device))
        self.A_log = nn.Parameter(torch.empty(self.n_head, device=config.device))
        self.D = nn.Parameter(torch.empty(self.n_head, device=config.device))
        self.norm = RMSNorm(self.n_inner, device=config.device)
        self.out_proj = nn.Linear(self.n_inner, self.n_embd, bias=False, device=config.device)

    def forward(self, u: Tensor, h: InferenceCache) -> (Tensor, InferenceCache):
        """
        Arguments
            u: (batch, seqlen, n_embd) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, n_embd) output
            h: updated inference cache after processing `u`
        """
        if self.mode == "sample":
            u = u[:, -1, :].unsqueeze(1)
            return self._step(u, h)

        init_conv, init_ssm = h

        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.n_inner,
                self.n_inner + 2 * self.n_state,
                self.n_head
            ],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        x_in = rearrange(xBC, "b l d -> b d l")  

        # Pad or truncate xBC seqlen to n_conv
        if self.n_conv > 1:
            left = init_conv[:, :, - (self.n_conv-1):] 
        else:
            left = init_conv[:, :, :0]
        x_cat = torch.cat([left, x_in], dim=-1) # (b, d, n_conv-1+L)
        x_conv = self.conv1d(x_cat)

        xBC = silu(rearrange(x_conv, "b d l -> b l d")) # (batch, seqlen, d_inner + 2* d_state)
        x, B, C = torch.split(
            xBC, [self.n_inner, self.n_state, self.n_state], dim=-1
        )
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        A = -torch.exp(self.A_log)  # (nheads,)
        y, new_ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.n_chunk,
            initial_states=init_ssm.unsqueeze(1), # (b, 1, h, p, n)
            device=self.device,
        )
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        new_conv_state = torch.cat([init_conv, x_in], dim=-1)[:, :, -self.n_conv:]  # (b, conv_dim, n_conv)
        return y, (new_conv_state, new_ssm_state)

    def _step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.n_inner,
                self.n_inner + 2 * self.n_state,
                self.n_head,
            ],
            dim=-1,
        )

        # Advance convolution input
        h[0].copy_(torch.roll(h[0], shifts=-1, dims=-1))
        h[0][:, :, -1] = xBC
        # Convolution step
        xBC = torch.sum(
            h[0] * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.n_inner, self.n_state, self.n_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSM step
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h[1].copy_(h[1] * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h[1], C)
        y = y + rearrange(self.D, "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h

    def initial_state(self, batch_size):
        return (
            torch.zeros(
                batch_size, self.n_inner + 2 * self.n_state, self.n_conv, device=self.device
            ),
            torch.zeros(
                batch_size, self.n_head, self.headdim, self.n_state, device=self.device
            ),
        )


def segsum(x: Tensor) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x, A, B, C, chunk_size, initial_states, device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0, f"{x.shape[1]} not chunkable by {chunk_size}"

    # Rearrange into chunks
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device = None):
        """Gated Root Mean Square Layer Normalization

        Paper: https://arxiv.org/abs/1910.07467
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * silu(z)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def silu(x):
    """Applies the Sigmoid Linear Unit (SiLU), element-wise.

    Define this manually since torch's version doesn't seem to work on MPS.
    """
    return x * F.sigmoid(x)

@dataclass
class GPTConfig:
    mode: str = "train"
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_inner: int = 1536 # n_embd * 2 ?
    n_conv: int = 4
    n_state: int = 384
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
        print(config)

        if config.block_type == "attention":
            backbone = nn.ModuleList([CsaBlock(config, i) for i in range(config.n_layer)])
        elif config.block_type == "mamba":
            backbone = nn.ModuleList([Mamba2(config, i) for i in range(config.n_layer)])
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
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for i, block in enumerate(self.transformer.backbone):
            x, state[i] = block(x, state[i])
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)                        
            B, T, V = logits.shape
            device, dtype = logits.device, logits.dtype

            ignore_index = -1
            loss_tok = F.cross_entropy(
                logits.view(-1, V),
                targets.view(-1),
                ignore_index=ignore_index,
                reduction="none",
            ).view(B, T)                                   

            weights = torch.tensor([1, 2, 4, 8], dtype=dtype, device=device)
            w_stream = weights[torch.arange(T, device=device) % 4]
            w_stream = w_stream.unsqueeze(0).expand(B, T)

            valid = (targets != ignore_index).to(dtype)
            w_stream = w_stream * valid

            # weighted mean
            denom = w_stream.sum().clamp_min(1e-12)
            loss = (loss_tok * w_stream).sum() / denom
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        def mean_by_quarter(x: torch.Tensor) -> torch.Tensor:
            quarters = torch.tensor_split(x, 4, dim=1)
            return torch.stack([q.mean(dim=1) for q in quarters], dim=1)

        def mean_by_mod4(x: torch.Tensor) -> torch.Tensor:
            return torch.stack([x[:, r::4].mean(dim=1) for r in range(4)], dim=1)

        if loss:
            print("mean by quarters", mean_by_quarter(loss))
            print("mean by mod4", mean_by_mod4(loss))

        return logits, state, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.backbone:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, state, _ = self(idx_cond, state)
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
