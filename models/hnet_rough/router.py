from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class RouterOutput:
    token_mask: torch.Tensor
    router_probs: torch.Tensor
    selected_probs: torch.Tensor
    cu_seqlens: list[int]


class RouterState(TypedDict):
    has_seen_token: torch.Tensor  # (B,) or (1, N)
    last_token: torch.Tensor  # (B, D) or (1, N, D)


# ---------------------------------------------------------------------------
# Routing strategies
# ---------------------------------------------------------------------------

class RoutingStrategy(nn.Module):
    """Base class for routing strategies.

    Subclasses implement a single method that maps hidden states to
    per-position boundary probabilities in [0, 1].
    """

    def prob_boundary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute boundary probabilities.

        Args:
            hidden_states: (B, L, D)

        Returns:
            (B, L) boundary probabilities in [0, 1].
        """
        raise NotImplementedError


class CosineSimRouting(RoutingStrategy):
    """Pairwise cosine similarity between adjacent positions.

    Position i gets the boundary probability derived from
    cos_sim(q(h[i-1]), k(h[i])). Position 0 defaults to 1.0 (always a
    boundary).
    """

    def __init__(self, d_model: int, *, init_mode: str = "default", device=None, dtype=None):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
        if init_mode == "identity":
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
            self.q_proj.weight._no_reinit = True
            self.k_proj.weight._no_reinit = True

    def prob_boundary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape
        if L <= 1:
            return torch.ones(B, L, device=hidden_states.device, dtype=torch.float32)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj(hidden_states[:, :-1]).float(), dim=-1),
            F.normalize(self.k_proj(hidden_states[:, 1:]).float(), dim=-1),
        )  # (B, L-1)

        # Position 0 defaults to 1.0 (always a boundary)
        default = torch.ones(B, 1, device=hidden_states.device, dtype=cos_sim.dtype)
        prob = torch.cat([default, ((1 - cos_sim) / 2).clamp(0, 1)], dim=1)  # (B, L)
        return prob


class LinearSigmoidRouting(RoutingStrategy):
    """Single linear projection producing per-position boundary logits."""

    def __init__(self, d_model: int, *, bias: bool = False, device=None, dtype=None):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=bias, device=device, dtype=dtype)

    def prob_boundary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden_states).squeeze(-1).float()  # (B, L)
        return torch.sigmoid(logits)


class MLPSigmoidRouting(RoutingStrategy):
    """Two-layer MLP with SwiGLU producing per-position boundary logits.

    Architecture: Linear(d_model, hidden*2) -> SwiGLU -> Linear(hidden, 1).
    """

    def __init__(
        self,
        d_model: int,
        *,
        expansion_factor: float = 2.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        hidden = int(d_model * expansion_factor)
        self.layer1 = nn.Linear(d_model, hidden * 2, bias=bias, device=device, dtype=dtype)
        self.layer2 = nn.Linear(hidden, 1, bias=bias, device=device, dtype=dtype)

    def prob_boundary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.layer1(hidden_states)
        a, b = x.chunk(2, dim=-1)
        x = b * F.silu(a)  # SwiGLU
        logits = self.layer2(x).squeeze(-1).float()  # (B, L)
        return torch.sigmoid(logits)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class Router(nn.Module):
    """Computes token boundaries using a pluggable routing strategy.

    Handles state management (cross-boundary pairs) and sequence-start
    overrides uniformly, delegating probability computation to the strategy.
    """

    def __init__(self, d_model: int, strategy: RoutingStrategy):
        super().__init__()
        self.d_model = d_model
        self.strategy = strategy

    def default_state(
        self,
        *batch_shape: int,
        last_token: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        inplace_state: Any = None,
        **kwargs,
    ) -> RouterState:
        device = next(self.parameters()).device if device is None else device
        dtype = next(self.parameters()).dtype if dtype is None else dtype
        assert inplace_state is None, "Inplace state is not supported for Router"

        if last_token is None:
            return RouterState(
                last_token=torch.zeros(batch_shape + (self.d_model,), device=device, dtype=dtype),
                has_seen_token=torch.zeros(batch_shape, device=device, dtype=torch.bool),
            )
        else:
            assert last_token.shape == batch_shape + (self.d_model,)
            return RouterState(
                last_token=last_token,
                has_seen_token=torch.ones(batch_shape, device=device, dtype=torch.bool),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        x_pack_kwargs: torch.Tensor,
        state: RouterState | None = None,
    ) -> tuple[RouterOutput, RouterState | None]:
        """Compute routing decisions for packed hidden states.

        Args:
            hidden_states: (1, L, D) packed input.
            state: Previous router state with (1, N, ...) leaf tensors.
            x_pack_kwargs: (1, N) per-sequence lengths.

        Returns:
            (RouterOutput, updated state or None).
        """
        device = hidden_states.device
        B, L, D = hidden_states.shape
        assert B == 1, "Packed input must have batch dimension 1"

        lens_cs = F.pad(x_pack_kwargs[0].cumsum(0), (1, 0)).long()

        if L == 0:
            return RouterOutput(
                token_mask=torch.zeros(B, L, device=device, dtype=torch.bool),
                router_probs=torch.zeros(B, L, 2, device=device, dtype=hidden_states.dtype),
                selected_probs=torch.zeros(B, L, 1, device=device, dtype=hidden_states.dtype),
                cu_seqlens=[0],
            ), state

        # -- Compute boundary probabilities via strategy -----------------------
        prob_boundary = self.strategy.prob_boundary(hidden_states)  # (1, L)

        # -- Override at sequence starts with cross-boundary pairs -------------
        if state is not None:
            last_token = state["last_token"].squeeze(0)  # (N, D)
            has_seen_token = state["has_seen_token"].squeeze(0)  # (N,)

            first_tokens = hidden_states[0, lens_cs[:-1]]  # (N, D)
            pairs = torch.stack([last_token, first_tokens], dim=1)  # (N, 2, D)
            cross_pb = self.strategy.prob_boundary(pairs)[:, 1]  # (N,)
            prob_boundary[0, lens_cs[:-1]] = torch.where(has_seen_token, cross_pb, 1.0)
        else:
            prob_boundary = prob_boundary.clone()
            prob_boundary[0, lens_cs[:-1]] = 1.0

        # -- Build outputs -----------------------------------------------------
        router_probs = torch.stack([1 - prob_boundary, prob_boundary], dim=-1)
        token_mask = prob_boundary > 0.5
        selected_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)

        router_output = RouterOutput(
            router_probs=router_probs,
            token_mask=token_mask,
            selected_probs=selected_probs,
            cu_seqlens=F.pad(token_mask.long().sum(dim=-1), (1, 0)).cumsum(dim=0).tolist(),
        )

        # -- Update state ------------------------------------------------------
        if state is not None:
            seq_end_positions = (lens_cs[1:] - 1).clamp(min=0)
            new_last_token = hidden_states[0, seq_end_positions]
            new_has_seen_token = x_pack_kwargs[0] > 0
            no_tokens = ~new_has_seen_token
            new_last_token[no_tokens] = last_token[no_tokens]
            new_has_seen_token = has_seen_token | new_has_seen_token
            state = RouterState(
                last_token=new_last_token.unsqueeze(0),
                has_seen_token=new_has_seen_token.unsqueeze(0),
            )

        return router_output, state

    def step(
        self, x: torch.Tensor, state: RouterState | None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RouterState]:
        """Single-step routing for autoregressive generation.

        Args:
            x: (B, D) current token.
            state: Required router state.

        Returns:
            (prob_boundary, is_token, selected_probs, new_state)
        """
        assert state is not None, "State is required for router step"

        last_token = state["last_token"]  # (B, D)
        has_seen_token = state["has_seen_token"]  # (B,)

        pairs = torch.stack([last_token, x], dim=1)  # (B, 2, D)
        prob_boundary = self.strategy.prob_boundary(pairs)[:, 1]  # (B,)
        prob_boundary = torch.where(has_seen_token, prob_boundary, torch.ones_like(prob_boundary))

        is_token = (prob_boundary > 0.5).detach()
        selected_probs = torch.where(is_token, prob_boundary, 1 - prob_boundary)

        new_state = RouterState(
            last_token=x.clone(),
            has_seen_token=torch.ones_like(has_seen_token),
        )
        return prob_boundary, is_token, selected_probs, new_state
