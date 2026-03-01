"""Stage: ordered container of Blocks with final norm."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from src.models.sequence.hnet.block import Block


class Stage(nn.Module):
    """Ordered container of Block modules with final RMSNorm.

    The stage is layout-agnostic: it accepts (1, T, D) packed or (B, L, D)
    unpacked input and simply iterates blocks. Packing/unpacking is the
    caller's responsibility.
    """

    def __init__(self, blocks: list[Block], d_model: int, track_flops: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d_model)
        self.track_flops = track_flops
        self.metrics: dict[str, Any] = {}

    def forward(
        self,
        x: Tensor,
        state: list | None = None,
        **kwargs: Any,
    ) -> tuple[Tensor, list | None, dict]:
        """Iterate blocks then apply final norm.

        Args:
            x: Input tensor — (1, T, D) packed or (B, L, D) unpacked.
            state: Per-block state list for inference, or None for training.
            **kwargs: Passed through to blocks (x_pack_kwargs, mask, etc.).

        Returns:
            (output, next_states_or_None, {})
        """
        self.metrics = {}
        if self.track_flops:
            self.metrics["flops"] = 0

        prev_states = [None] * len(self.blocks) if state is None else state
        next_states = []
        for block, prev_state in zip(self.blocks, prev_states):
            x, block_state = block(x, state=prev_state, **kwargs)
            next_states.append(block_state)
            if self.track_flops:
                self.metrics["flops"] += block.layer.get_flops(x, **kwargs)

        x = self.norm(x)
        return x, next_states if state is not None else None, {}

    def step(self, x: Tensor, state: list, **kwargs) -> tuple[Tensor, list]:
        next_states = []
        for block, prev_state in zip(self.blocks, state):
            x, s = block.step(x, state=prev_state, **kwargs)
            next_states.append(s)
        x = self.norm(x)
        return x, next_states

    def default_state(self, *batch_shape, **kwargs) -> list:
        return [block.default_state(*batch_shape, **kwargs) for block in self.blocks]

    def n_residual(self) -> int:
        return len(self.blocks)
