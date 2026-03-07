"""
Grouped Rotation Attention: sphere-preserving self-attention via Givens rotations.

n_attn_heads shared Q/K attention heads each drive n_rot_planes = (n_embd // 2) // n_attn_heads
rotation planes. The attention-weighted angles are applied as 2D rotations in each plane.
No residual around attention (rotation IS the residual: zero angle = identity = skip).
Standard residual + RMSNorm around MLP, then normalize to unit sphere.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.model import ModuleList, MLP


@dataclass
class RotationConfig:
    n_embd: int
    n_attn_heads: int
    d_k: int        # QK head dimension per attention head
    block_size: int
    bias: bool
    dropout: float


class RotationAttention(nn.Module):
    """Grouped rotation attention.

    n_attn_heads attention heads share Q/K projections. Each head drives
    n_rot_planes = (n_embd // 2) // n_attn_heads rotation planes via
    per-plane scalar value projections.
    """

    def __init__(self, config: RotationConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_attn_heads = config.n_attn_heads
        self.d_k = config.d_k
        self.n_rot_planes = (config.n_embd // 2) // config.n_attn_heads
        self.dropout = config.dropout

        assert config.n_embd % (2 * config.n_attn_heads) == 0

        # Fused Q/K: (n_embd) -> (2 * n_attn_heads * d_k)
        self.W_QK = nn.Linear(config.n_embd, 2 * config.n_attn_heads * config.d_k, bias=False)
        # Value: (n_embd) -> (n_embd // 2), one scalar per rotation plane
        self.W_V = nn.Linear(config.n_embd, config.n_embd // 2, bias=False)
        # Orthogonal output projection (Householder parameterization) — preserves sphere
        self.W_out = torch.nn.utils.parametrizations.orthogonal(
            nn.Linear(config.n_embd, config.n_embd, bias=False)
        )
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, state: tuple[Tensor, Tensor], positions=None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T, C = x.size()
        H, d_k, R = self.n_attn_heads, self.d_k, self.n_rot_planes

        # Q/K projections: (B, T, 2*H*d_k) -> (B, H, T, d_k)
        qk = self.W_QK(x)
        q, k = qk.split(H * d_k, dim=-1)
        q = q.view(B, T, H, d_k).transpose(1, 2)
        k = k.view(B, T, H, d_k).transpose(1, 2)

        # Value projection: (B, T, n_embd//2) -> (B, H, T, R)
        theta = self.W_V(x).view(B, T, H, R).transpose(1, 2)

        # Cache: append new K and theta
        cached_k, cached_theta = state
        offset = cached_k.size(2)
        k = torch.cat([cached_k, k], dim=2)
        theta = torch.cat([cached_theta, theta], dim=2)

        # Aggregate angles via attention: (B, H, T_new, R)
        neg_inf = torch.finfo(torch.float32).min
        causal = torch.triu(torch.full((T, T), neg_inf, device=x.device, dtype=torch.float32), diagonal=1)
        attn_mask = torch.cat([
            torch.zeros(T, offset, device=x.device, dtype=torch.float32),
            causal,
        ], dim=1)

        phi = F.scaled_dot_product_attention(
            q, k, theta,
            attn_mask=None if self.training else attn_mask,
            is_causal=True if self.training else False,
            dropout_p=self.dropout if self.training else 0,
        )

        # Reshape phi to (B, T, n_embd//2)
        phi = phi.transpose(1, 2).reshape(B, T, C // 2)

        # Apply 2D rotations
        cos_phi = torch.cos(phi).unsqueeze(-1)  # (B, T, n_embd//2, 1)
        sin_phi = torch.sin(phi).unsqueeze(-1)

        x_pairs = x.view(B, T, C // 2, 2)
        x0 = x_pairs[..., 0:1]
        x1 = x_pairs[..., 1:2]
        y = torch.cat([cos_phi * x0 - sin_phi * x1,
                        sin_phi * x0 + cos_phi * x1], dim=-1)
        y = y.reshape(B, T, C)
        y = self.W_out(y)

        return y, (k, theta)


class RotationBlock(nn.Module):
    """Rotation attention (no residual) + RMSNorm + MLP (with residual) + normalize."""

    def __init__(self, config: RotationConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_attn_heads = config.n_attn_heads
        self.d_k = config.d_k
        self.n_rot_planes = (config.n_embd // 2) // config.n_attn_heads
        self.block_size = config.block_size

        self.attn = RotationAttention(config, layer_idx)
        self.mlp = MLP(config.n_embd, bias=config.bias, dropout=config.dropout)

    def forward(self, x: Tensor, state, positions=None) -> tuple[Tensor, tuple]:
        # Rotation attention (no residual — rotation absorbs it)
        x, state = self.attn(x, state, positions)
        # MLP with residual + RMSNorm, then re-normalize to sphere
        x = F.normalize(x + self.mlp(x), dim=-1)
        return x, state

    def initial_state(self, batch_size: int):
        device = next(self.parameters()).device
        empty_k = torch.zeros(batch_size, self.n_attn_heads, 0, self.d_k, device=device)
        empty_theta = torch.zeros(batch_size, self.n_attn_heads, 0, self.n_rot_planes, device=device)
        return (empty_k, empty_theta)

    def flops_per_fwdbwd(self):
        N = sum(p.numel() for p in self.parameters())
        H, d_k, T = self.n_attn_heads, self.d_k, self.block_size
        flops_per_token = 6 * N + 12 * H * d_k * T
        return flops_per_token * T


class NormalizeInput(nn.Module):
    """Wraps a backbone to normalize input embeddings onto the unit sphere."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x, state):
        return self.backbone(F.normalize(x, dim=-1), state)

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()


def make_rotation_backbone(*, n_layer, n_embd, n_attn_heads=12, d_k=32, block_size, bias, dropout):
    """Build a NormalizeInput-wrapped ModuleList of RotationBlocks."""
    config = RotationConfig(
        n_embd=n_embd,
        n_attn_heads=n_attn_heads,
        d_k=d_k,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
    )
    blocks = ModuleList([RotationBlock(config, i) for i in range(n_layer)])
    return NormalizeInput(blocks)


# ---------------------------------------------------------------------------
# Kronecker Rotation Attention
# ---------------------------------------------------------------------------

@dataclass
class KroneckerRotationConfig:
    n_embd: int
    n_attn_heads: int
    d_k: int              # QK head dimension
    factors: tuple[int, ...]  # product must == n_embd
    block_size: int
    bias: bool
    dropout: float


def _skew_exp(params: Tensor, size: int) -> Tensor:
    """Rotation matrices from skew-symmetric parameters via matrix_exp.

    params: (..., size*(size-1)//2)  — upper-triangle entries
    returns: (..., size, size)       — rotation matrices
    """
    orig_dtype = params.dtype
    params = params.float()
    batch_shape = params.shape[:-1]
    mat = params.new_zeros(*batch_shape, size, size)
    rows, cols = torch.triu_indices(size, size, offset=1)
    mat[..., rows, cols] = params
    S = mat - mat.transpose(-2, -1)
    return torch.matrix_exp(S).to(orig_dtype)


class KroneckerRotationAttention(nn.Module):
    """Kronecker-factored rotation attention.

    Each position produces skew-symmetric params for small factor matrices.
    Exponentiating gives rotation matrices; their Kronecker product acts on
    the input without materializing the full d×d rotation.
    """

    def __init__(self, config: KroneckerRotationConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_attn_heads = config.n_attn_heads
        self.d_k = config.d_k
        self.factors = config.factors
        self.dropout = config.dropout

        prod = 1
        for f in config.factors:
            prod *= f
        assert prod == config.n_embd, (
            f"Product of factors {config.factors} = {prod} != n_embd={config.n_embd}"
        )

        self.factor_params = tuple(f * (f - 1) // 2 for f in config.factors)
        self.d_v = sum(self.factor_params)
        # Pad d_v to next power of 2 so flash/mem-efficient SDPA kernels can be used
        self.d_v_padded = 1 << (self.d_v - 1).bit_length()

        # Q/K projection (all heads fused)
        self.W_QK = nn.Linear(config.n_embd, 2 * config.n_attn_heads * config.d_k, bias=False)
        # V projection: each head produces d_v skew-symmetric params
        self.W_V = nn.Linear(config.n_embd, config.n_attn_heads * self.d_v, bias=False)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor], positions=None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        B, T, C = x.size()
        H, d_k, d_v = self.n_attn_heads, self.d_k, self.d_v

        # --- Q / K / V projections ---
        qk = self.W_QK(x)
        q, k = qk.split(H * d_k, dim=-1)
        q = q.view(B, T, H, d_k).transpose(1, 2)
        k = k.view(B, T, H, d_k).transpose(1, 2)

        v = self.W_V(x).view(B, T, H, d_v).transpose(1, 2)
        # Pad v to power-of-2 for flash attention compatibility
        if self.d_v_padded != d_v:
            v = F.pad(v, (0, self.d_v_padded - d_v))

        # --- KV cache ---
        cached_k, cached_v = state
        offset = cached_k.size(2)
        k = torch.cat([cached_k, k], dim=2)
        v = torch.cat([cached_v, v], dim=2)

        # --- Attention: aggregate rotation params ---
        neg_inf = torch.finfo(torch.float32).min
        causal = torch.triu(
            torch.full((T, T), neg_inf, device=x.device, dtype=torch.float32),
            diagonal=1,
        )
        attn_mask = torch.cat([
            torch.zeros(T, offset, device=x.device, dtype=torch.float32),
            causal,
        ], dim=1)

        # (B, H, T_new, d_v_padded)
        rot_params = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if self.training else attn_mask,
            is_causal=True if self.training else False,
            dropout_p=self.dropout if self.training else 0,
        )
        # Slice off padding
        if self.d_v_padded != d_v:
            rot_params = rot_params[..., :d_v]

        # Sum across heads in Lie algebra space: (B, T, d_v)
        rot_params = rot_params.transpose(1, 2).sum(dim=2)

        # --- Build rotation matrices and apply via Kronecker product ---
        param_splits = rot_params.split(list(self.factor_params), dim=-1)
        y = x.view(B, T, *self.factors)
        for i, (p, f) in enumerate(zip(param_splits, self.factors)):
            R = _skew_exp(p, f)  # (B, T, f, f)
            # Contract R along factor axis i (position 2+i in y).
            # Move target axis to position 2, matmul, move back.
            y = y.movedim(2 + i, 2)                       # target axis -> pos 2
            shape = y.shape
            y = (R @ y.reshape(B, T, f, -1)).reshape(shape)
            y = y.movedim(2, 2 + i)                        # pos 2 -> target axis
        y = y.reshape(B, T, C)

        return y, (k, v)


class KroneckerRotationBlock(nn.Module):
    """Kronecker rotation attention (no residual) + MLP (with residual) + normalize."""

    def __init__(self, config: KroneckerRotationConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_attn_heads = config.n_attn_heads
        self.d_k = config.d_k
        self.factors = config.factors
        self.block_size = config.block_size
        self.d_v = sum(f * (f - 1) // 2 for f in config.factors)
        self.d_v_padded = 1 << (self.d_v - 1).bit_length()

        self.attn = KroneckerRotationAttention(config, layer_idx)
        self.mlp = MLP(config.n_embd, bias=config.bias, dropout=config.dropout)

    def forward(self, x: Tensor, state, positions=None) -> tuple[Tensor, tuple]:
        attn_out, state = self.attn(x, state, positions)
        x = x + attn_out
        x = F.normalize(x + self.mlp(x), dim=-1)
        return x, state

    def initial_state(self, batch_size: int):
        device = next(self.parameters()).device
        empty_k = torch.zeros(batch_size, self.n_attn_heads, 0, self.d_k, device=device)
        empty_v = torch.zeros(batch_size, self.n_attn_heads, 0, self.d_v_padded, device=device)
        return (empty_k, empty_v)

    def flops_per_fwdbwd(self):
        N = sum(p.numel() for p in self.parameters())
        H, d_k, T = self.n_attn_heads, self.d_k, self.block_size
        flops_per_token = 6 * N + 12 * H * d_k * T
        return flops_per_token * T


def make_kronecker_rotation_backbone(
    *, n_layer, n_embd, n_attn_heads, d_k, factors, block_size, bias, dropout,
):
    """Build a NormalizeInput-wrapped ModuleList of KroneckerRotationBlocks."""
    config = KroneckerRotationConfig(
        n_embd=n_embd,
        n_attn_heads=n_attn_heads,
        d_k=d_k,
        factors=factors,
        block_size=block_size,
        bias=bias,
        dropout=dropout,
    )
    blocks = ModuleList([KroneckerRotationBlock(config, i) for i in range(n_layer)])
    return NormalizeInput(blocks)
