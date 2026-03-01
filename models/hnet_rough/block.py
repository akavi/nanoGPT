"""Simple prenorm residual block for hnet."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class Block(nn.Module):
    """Prenorm residual block wrapping a single leaf layer (SSD, FFN, or SA).
    Replicates the prenorm + fp32-residual path of SequenceResidualBlock
    without the full config machinery.
    """

    def __init__(self, layer: nn.Module, d_model: int, residual_in_fp32: bool = True):
        super().__init__()
        self.layer = layer
        self.norm = nn.RMSNorm(d_model)
        self.residual_in_fp32 = residual_in_fp32

    def forward(self, x: Tensor, state: Any = None, **kwargs) -> tuple[Tensor, Any]:
        residual = x
        y = self.norm(x)
        y, state = self.layer(y, state=state, **kwargs)
        dtype = y.dtype
        if self.residual_in_fp32:
            y = (residual.float() + y.float()).to(dtype)
        else:
            y = residual + y
        return y, state

    def step(self, x: Tensor, state: Any = None, **kwargs) -> tuple[Tensor, Any]:
        residual = x
        y = self.norm(x)
        y, state = self.layer.step(y, state=state, **kwargs)
        dtype = y.dtype
        if self.residual_in_fp32:
            y = (residual.float() + y.float()).to(dtype)
        else:
            y = residual + y
        return y, state

    def default_state(self, *batch_shape, **kwargs):
        return self.layer.default_state(*batch_shape, **kwargs)
