from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Routing strategies
# ---------------------------------------------------------------------------

class RoutingStrategy(nn.Module):
    """Base class for routing strategies.

    Subclasses implement a single method that maps hidden states to
    per-position boundary probabilities in [0, 1].
    """

    def prob_boundary(self, hidden_states: Tensor) -> Tensor:
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

    def __init__(self, d_model: int, *, init_mode: str = "orthogonal"):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        if init_mode == "identity":
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
        elif init_mode == "orthogonal":
            nn.init.orthogonal_(self.q_proj.weight)
            nn.init.orthogonal_(self.k_proj.weight)
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def prob_boundary(self, hidden_states: Tensor) -> Tensor:
        B, L, D = hidden_states.shape
        if L <= 1:
            return torch.ones(B, L, device=hidden_states.device, dtype=torch.float32)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj(hidden_states[:, :-1]).float(), dim=-1),
            F.normalize(self.k_proj(hidden_states[:, 1:]).float(), dim=-1),
        )  # (B, L-1)

        default = torch.ones(B, 1, device=hidden_states.device, dtype=cos_sim.dtype)
        prob = torch.cat([default, ((1 - cos_sim) / 2).clamp(0, 1)], dim=1)  # (B, L)
        return prob


class LinearSigmoidRouting(RoutingStrategy):
    """Single linear projection producing per-position boundary logits."""

    def __init__(self, d_model: int, *, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=bias)

    def prob_boundary(self, hidden_states: Tensor) -> Tensor:
        logits = self.proj(hidden_states).squeeze(-1).float()  # (B, L)
        return torch.sigmoid(logits)


class MLPSigmoidRouting(RoutingStrategy):
    """Two-layer MLP with SwiGLU producing per-position boundary logits.

    Architecture: Linear(d_model, hidden*2) -> SwiGLU -> Linear(hidden, 1).
    """

    def __init__(self, d_model: int, *, expansion_factor: float = 2.0, bias: bool = True):
        super().__init__()
        hidden = int(d_model * expansion_factor)
        self.layer1 = nn.Linear(d_model, hidden * 2, bias=bias)
        self.layer2 = nn.Linear(hidden, 1, bias=bias)

    def prob_boundary(self, hidden_states: Tensor) -> Tensor:
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

    def forward(
        self,
        hidden_states: Tensor,  # (B, L, D)
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None, dict[str, float]]:
        """Compute routing decisions.

        Args:
            hidden_states: (B, L, D)
            state: (last_token (B,D), has_seen_token (B,)) or None

        Returns:
            (prob, token_mask, new_state, metrics)
        """
        B, L, D = hidden_states.shape

        prob = self.strategy.prob_boundary(hidden_states)  # (B, L)

        # Handle cross-boundary state from previous forward call
        if state is not None:
            last_token, has_seen_token = state
            pairs = torch.stack([last_token, hidden_states[:, 0]], dim=1)  # (B, 2, D)
            cross_prob = self.strategy.prob_boundary(pairs)[:, 1]  # (B,)
            prob = prob.clone()
            prob[:, 0] = torch.where(has_seen_token, cross_prob, torch.ones_like(cross_prob))
        else:
            prob = prob.clone()
            prob[:, 0] = 1.0  # First position is always a boundary

        token_mask = prob > 0.5

        new_state = (
            hidden_states[:, -1].detach().clone(),
            torch.ones(B, device=hidden_states.device, dtype=torch.bool),
        )

        metrics: dict[str, float] = {}
        if isinstance(self.strategy, CosineSimRouting):
            # Recompute cos_sim for diagnostics (cheap relative to forward)
            if L > 1:
                cos_sim = torch.einsum(
                    "b l d, b l d -> b l",
                    F.normalize(self.strategy.q_proj(hidden_states[:, :-1]).float(), dim=-1),
                    F.normalize(self.strategy.k_proj(hidden_states[:, 1:]).float(), dim=-1),
                )
                metrics['cos_sim_mean'] = cos_sim.mean().item()
                metrics['cos_sim_std'] = cos_sim.std().item()

        return prob, token_mask, new_state, metrics

    def initial_state(self, batch_size: int) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.d_model, device=device),
            torch.zeros(batch_size, device=device, dtype=torch.bool),
        )
