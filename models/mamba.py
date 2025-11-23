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

@dataclass
class MambaConfig:
    n_state: int
    n_head: int = 12
    n_embd: int = 768
    n_inner: int = 1536 # n_embd * 2 ?
    n_conv: int = 4
    n_chunk: int = 64
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_decay: bool = False
    device: str = "cuda"
    mode: str = "train"

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

