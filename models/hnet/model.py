"""
H-Net: Hierarchical model with dynamic chunking (data-dependent token boundaries).

Architecture at each level:
    input -> Encoder (SSD blocks) -> Router (cosine-sim boundary detection)
                                        |
                         Tokenizer (compact boundary positions)
                                        |
                         Main Stage (deeper/wider model on compressed seq)
                                        |
                         DeTokenizer (EMA upsample + residual connection)
                                        |
                         Decoder (SSD blocks) -> output

Everything in one file, reusing Mamba2/ssd from models/mamba.py.
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.mamba import RMSNorm
from models.hnet.router import Router


# ---------------------------------------------------------------------------
# STE (straight-through estimator)
# ---------------------------------------------------------------------------

class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste(x):
    """Forward: returns ones_like(x). Backward: passes gradient through."""
    return _STE.apply(x)


# ---------------------------------------------------------------------------
# EMA via parallel scan in [0, 1]
# ---------------------------------------------------------------------------

def ema_scan(
    x: Tensor,              # (B, L, D) — input values
    decay: Tensor,           # (B, L) — decay factors in [0, 1]
    initial_state: Tensor,   # (B, D)
) -> Tensor:
    """Sequential EMA: h[t] = decay[t] * h[t-1] + (1 - decay[t]) * x[t].

    O(L) work, O(1) autograd memory (only saves h and inputs, not O(log L)
    intermediates like the Hillis-Steele parallel scan).
    """
    B, L, D = x.shape
    # a[t] = decay[t], b[t] = (1 - decay[t]) * x[t]
    a = decay.unsqueeze(-1).expand(B, L, D)       # (B, L, D)
    b = (1 - decay).unsqueeze(-1).expand(B, L, D) * x  # (B, L, D)

    # Fold initial state into the first position:
    # h[0] = decay[0] * init + (1 - decay[0]) * x[0]
    #       = a[0] * init + b[0]
    b = torch.cat([b[:, :1] + a[:, :1] * initial_state.unsqueeze(1), b[:, 1:]], dim=1)

    stride = 1
    while stride < L:
        b = torch.cat([b[:, :stride],
                        a[:, stride:] * b[:, :-stride] + b[:, stride:]], dim=1)
        a = torch.cat([a[:, :stride],
                        a[:, stride:] * a[:, :-stride]], dim=1)
        stride *= 2

    return b




# ---------------------------------------------------------------------------
# Tokenizer / DeTokenizer
# ---------------------------------------------------------------------------

class Tokenizer(nn.Module):
    """Compacts hidden states at boundary positions.

    For each batch element, selects positions where token_mask is True.
    Pads to max token count across the batch.
    """

    def forward(
        self,
        hidden_states: Tensor,  # (B, L, D)
        token_mask: Tensor,     # (B, L) bool
        positions: Tensor | None = None,  # (L,) original position indices
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Returns:
            chunked: (B, M, D) where M = max tokens in any batch element
            counts: (B,) number of tokens per batch element
            selected_positions: (M,) position indices of selected tokens (from batch 0)
        """
        B, L, D = hidden_states.shape
        counts = token_mask.sum(dim=1)  # (B,)
        M = int(counts.max().item())

        if M == 0:
            return hidden_states.new_zeros(B, 0, D), counts, None

        chunked = hidden_states.new_zeros(B, M, D)
        for b in range(B):
            selected = hidden_states[b, token_mask[b]]  # (n_tokens, D)
            chunked[b, :selected.shape[0]] = selected

        # Compute selected positions from batch 0 (positions are shared across batch)
        if positions is not None:
            sel_pos = positions[token_mask[0]]  # (n_tokens_b0,)
            # Pad to M
            selected_positions = torch.zeros(M, dtype=torch.long, device=positions.device)
            selected_positions[:sel_pos.shape[0]] = sel_pos
        else:
            selected_positions = None

        return chunked, counts, selected_positions


class DeTokenizer(nn.Module):
    """EMA-based upsampling + STE weighting + residual add-back.

    Runs EMA over the compacted chunk states, then broadcasts back
    to original sequence length via cumulative chunk indexing.
    """

    def __init__(self, d_model: int, norm: bool = False):
        super().__init__()
        self.d_model = d_model
        self.norm_main = RMSNorm(d_model) if norm else None
        self.norm_res = RMSNorm(d_model) if norm else None
        self.register_buffer('_device_buf', torch.empty(0), persistent=False)

    def forward(
        self,
        hidden_states: Tensor,  # (B, M, D) — compacted chunk states
        residual: Tensor,       # (B, L, D) — pre-router residual
        token_mask: Tensor,     # (B, L) bool
        prob: Tensor,           # (B, L) boundary probabilities
        counts: Tensor,         # (B,) tokens per batch element
        state: Tensor | None = None,  # (B, D) previous EMA value
        detach_residual: bool = False,
        residual_drop: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            output: (B, L, D)
            new_state: (B, D) last EMA value per batch element
        """
        B, L, D = residual.shape
        M = hidden_states.shape[1]
        device = hidden_states.device

        if detach_residual:
            residual = residual.detach()

        if M == 0:
            contribution = state.unsqueeze(1).expand(B, L, D)
            output = (residual.float() + contribution.float()).to(residual.dtype)
            detok_metrics = {'residual_norm': residual.float().norm().item(), 'main_norm': contribution.float().norm().item()}
            return output, state, detok_metrics

        # Build chunk probs: gather boundary probs at token positions
        chunk_probs = prob.new_zeros(B, M)
        valid_mask = torch.arange(M, device=device)[None, :] < counts[:, None]  # (B, M)
        for b in range(B):
            selected_probs = prob[b, token_mask[b]]  # (n_tokens,)
            chunk_probs[b, :selected_probs.shape[0]] = selected_probs

        # decay = 1 - p: high boundary prob → low decay (forget the past)
        decay = (1 - chunk_probs).clamp(0, 1)  # (B, M)

        # Run EMA: h[t] = decay[t] * h[t-1] + (1 - decay[t]) * x[t]
        ema_out = ema_scan(hidden_states, decay, state)
        # Zero out invalid positions
        ema_out = ema_out * valid_mask.unsqueeze(-1).to(ema_out.dtype)

        # Broadcast back to original sequence length
        # chunk_idx[b, i] = index of chunk that position i belongs to
        chunk_idx = torch.cumsum(token_mask.long(), dim=1) - 1  # (B, L)
        chunk_idx_clamped = chunk_idx.clamp(min=0, max=M - 1)
        valid_positions = chunk_idx >= 0  # (B, L)

        # Gather EMA output at chunk positions
        batch_idx = torch.arange(B, device=device)[:, None]
        long_states = ema_out[batch_idx, chunk_idx_clamped]  # (B, L, D)
        long_states = long_states * valid_positions.unsqueeze(-1).to(long_states.dtype)

        # STE weighting: forward=1, backward=prob
        router_probs_stacked = torch.stack([1 - prob, prob], dim=-1)  # (B, L, 2)
        coef = torch.max(router_probs_stacked, dim=-1).values  # (B, L)
        coef = ste(coef).to(long_states.dtype)
        long_states = long_states * coef.unsqueeze(-1)

        # Residual add-back (in fp32)
        out_dtype = long_states.dtype
        if self.norm_main is not None:
            long_states = self.norm_main(long_states)
        if self.training and residual_drop > 0:
            keep = torch.bernoulli(torch.full((B, L, 1), 1 - residual_drop, device=device))
            residual = residual * keep
            output = (residual.float() + long_states.float()).to(out_dtype)
        else:
            if self.norm_res is not None:
                residual = self.norm_res(residual)
            output = (residual.float() + long_states.float()).to(out_dtype)

        # Compute new state: last EMA output per batch element
        new_state = residual.new_zeros(B, D)
        for b in range(B):
            c = int(counts[b].item())
            if c > 0:
                new_state[b] = ema_out[b, c - 1]
            elif state is not None:
                new_state[b] = state[b]

        detok_metrics = {'residual_norm': residual.float().norm().item(), 'main_norm': long_states.float().norm().item()}
        return output, new_state, detok_metrics

    def initial_state(self, batch_size):
        device = self._device_buf.device
        return torch.zeros(batch_size, self.d_model, device=device)


class CrossAttentionDeTokenizer(nn.Module):
    """Cross-attention upsampling + residual add-back.

    Each full-resolution position attends to all compressed tokens,
    allowing content-based routing independent of sequence order.
    Drop-in replacement for DeTokenizer.
    """

    def __init__(self, d_model: int, n_head: int = 4, norm: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_main = RMSNorm(d_model) if norm else None
        self.norm_res = RMSNorm(d_model) if norm else None
        self.register_buffer('_device_buf', torch.empty(0), persistent=False)

    def forward(
        self,
        hidden_states: Tensor,  # (B, M, D) — compacted chunk states
        residual: Tensor,       # (B, L, D) — pre-router residual
        token_mask: Tensor,     # (B, L) bool — unused, kept for interface
        prob: Tensor,           # (B, L) — unused, kept for interface
        counts: Tensor,         # (B,) tokens per batch element
        state: Tensor | None = None,  # (B, D) previous value
        detach_residual: bool = False,
        residual_drop: float = 0.0,
    ) -> tuple[Tensor, Tensor, dict]:
        B, L, D = residual.shape
        M = hidden_states.shape[1]
        device = hidden_states.device

        if detach_residual:
            residual = residual.detach()

        if M == 0:
            contribution = state.unsqueeze(1).expand(B, L, D)
            output = (residual.float() + contribution.float()).to(residual.dtype)
            return output, state, {'residual_norm': residual.float().norm().item(), 'main_norm': contribution.float().norm().item()}

        hs = D // self.n_head
        q = self.q_proj(residual).view(B, L, self.n_head, hs).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, M, self.n_head, hs).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, M, self.n_head, hs).transpose(1, 2)

        # Mask invalid compressed positions
        valid_mask = torch.arange(M, device=device)[None, :] < counts[:, None]  # (B, M)
        attn_bias = torch.where(
            valid_mask[:, None, None, :].expand(B, self.n_head, L, M),
            torch.zeros(1, device=device, dtype=torch.float32),
            torch.full((1,), float('-inf'), device=device, dtype=torch.float32),
        )

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        long_states = self.out_proj(attn_out.transpose(1, 2).contiguous().view(B, L, D))

        # Residual add-back (in fp32)
        out_dtype = long_states.dtype
        if self.norm_main is not None:
            long_states = self.norm_main(long_states)
        if self.training and residual_drop > 0:
            keep = torch.bernoulli(torch.full((B, L, 1), 1 - residual_drop, device=device))
            residual = residual * keep
            output = (residual.float() + long_states.float()).to(out_dtype)
        else:
            if self.norm_res is not None:
                residual = self.norm_res(residual)
            output = (residual.float() + long_states.float()).to(out_dtype)

        # State: last valid compressed token per batch element
        new_state = residual.new_zeros(B, D)
        for b in range(B):
            c = int(counts[b].item())
            if c > 0:
                new_state[b] = hidden_states[b, c - 1]
            elif state is not None:
                new_state[b] = state[b]

        return output, new_state, {'residual_norm': residual.float().norm().item(), 'main_norm': long_states.float().norm().item()}

    def initial_state(self, batch_size):
        device = self._device_buf.device
        return torch.zeros(batch_size, self.d_model, device=device)


# ---------------------------------------------------------------------------
# Block / Stage
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Prenorm residual block: RMSNorm -> layer -> residual add."""

    def __init__(self, layer: nn.Module, d_model: int):
        super().__init__()
        self.layer = layer
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor, state) -> tuple[Tensor, object]:
        y = self.norm(x)
        y, new_state = self.layer(y, state)
        return (x.float() + y.float()).to(y.dtype), new_state

    def initial_state(self, batch_size):
        return self.layer.initial_state(batch_size)

    def flops_per_fwdbwd(self):
        return self.layer.flops_per_fwdbwd()


class Stage(nn.Module):
    """Ordered list of Blocks + final RMSNorm."""

    def __init__(self, blocks: list, d_model: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(d_model)

    def forward(self, x: Tensor, state: list, positions: Tensor | None = None) -> tuple[Tensor, list]:
        new_state = []
        for block, s in zip(self.blocks, state):
            x, s_new = block(x, s, positions=positions)
            new_state.append(s_new)
        x = self.final_norm(x)
        return x, new_state

    def initial_state(self, batch_size):
        return [block.initial_state(batch_size) for block in self.blocks]

    def flops_per_fwdbwd(self):
        return sum(block.flops_per_fwdbwd() for block in self.blocks)


# ---------------------------------------------------------------------------
# MLP (SwiGLU)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """SwiGLU MLP: Linear(d, hidden*2) -> SwiGLU -> Linear(hidden, d)."""

    def __init__(self, d_model: int, d_inner: int | None = None, bias: bool = False):
        super().__init__()
        d_inner = d_inner or int(d_model * 2.667)  # ~8/3 expansion
        self.w1 = nn.Linear(d_model, d_inner * 2, bias=bias)
        self.w2 = nn.Linear(d_inner, d_model, bias=bias)

    def forward(self, x: Tensor, state) -> tuple[Tensor, None]:
        gate_and_val = self.w1(x)
        gate, val = gate_and_val.chunk(2, dim=-1)
        x = val * F.silu(gate)
        return self.w2(x), state

    def initial_state(self, batch_size):
        return None

    def flops_per_fwdbwd(self):
        return 0


# ---------------------------------------------------------------------------
# HNet (one level of the hierarchy)
# ---------------------------------------------------------------------------

class AnisotropicStack(nn.Module):
    """Dimension-changing stack: pre → proj_up → inner → proj_down → post.

    No routing or sequence compression — all stages see the same sequence length.
    Dimensions change via linear projections between stages.
    """

    def __init__(
        self,
        d_model: int,
        pre_stage: Stage,
        main_stage: nn.Module,  # Stage, HNet, or AnisotropicStack
        post_stage: Stage,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        inner_lr_ratio: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pre_stage = pre_stage
        self.main_stage = main_stage
        self.post_stage = post_stage
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.inner_lr_ratio = inner_lr_ratio

    def forward(self, x: Tensor, state: dict, positions: Tensor | None = None, train_step: tuple[int, int] | None = None) -> tuple[Tensor, dict, Tensor, dict[str, float]]:
        # 1. Pre-stage
        x, pre_state = self.pre_stage(x, state['pre'], positions=positions)

        # 2. Project up to inner dimension
        x = self.up_proj(x)

        # 3. Main stage (inner dimension)
        if isinstance(self.main_stage, (HNet, AnisotropicStack)):
            x, main_state, aux_loss, metrics = self.main_stage(x, state['main'], positions=positions, train_step=train_step)
        else:
            x, main_state = self.main_stage(x, state['main'], positions=positions)
            aux_loss = torch.tensor(0.0, device=x.device)
            metrics = {}

        # 4. Project down to outer dimension
        x = self.down_proj(x)

        # 5. Post-stage
        x, post_state = self.post_stage(x, state['post'], positions=positions)

        new_state = {'pre': pre_state, 'main': main_state, 'post': post_state}
        return x, new_state, aux_loss, metrics

    def optim_groups(self, lr_scale: float = 1.0) -> list[dict]:
        from utils import decay_nodecay_groups
        outer_params = list(self.pre_stage.parameters()) + \
                       list(self.post_stage.parameters()) + \
                       list(self.up_proj.parameters()) + \
                       list(self.down_proj.parameters())
        groups = decay_nodecay_groups(outer_params, lr_scale)

        inner_scale = lr_scale / self.inner_lr_ratio
        if isinstance(self.main_stage, (HNet, AnisotropicStack)):
            groups += self.main_stage.optim_groups(lr_scale=inner_scale)
        else:
            groups += decay_nodecay_groups(self.main_stage.parameters(), inner_scale)
        return groups

    def initial_state(self, batch_size):
        return {
            'pre': self.pre_stage.initial_state(batch_size),
            'main': self.main_stage.initial_state(batch_size),
            'post': self.post_stage.initial_state(batch_size),
        }

    def flops_per_fwdbwd(self):
        return self.pre_stage.flops_per_fwdbwd() + \
               self.main_stage.flops_per_fwdbwd() + \
               self.post_stage.flops_per_fwdbwd()


class HNet(nn.Module):
    """One level of the H-Net hierarchy. Recursable.

    pre_stage -> router -> tokenizer -> main_stage -> detokenizer -> post_stage

    main_stage can be either a Stage (leaf) or another HNet (nested).
    """

    def __init__(
        self,
        d_model: int,
        pre_stage: Stage,
        main_stage: nn.Module,  # Stage or HNet
        post_stage: Stage,
        router: Router,
        detokenizer: DeTokenizer,
        residual_proj: nn.Linear,
        pad_parameter: nn.Parameter | None,
        target_ratio: float = 4.0,
        inner_lr_ratio: float | None = None,
        detach_residual: bool = False,
        residual_drop_fn: Callable[[float], float] = lambda _: 0.0,
        ratio_override_fn: Callable[[float], float] | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.pre_stage = pre_stage
        self.main_stage = main_stage
        self.post_stage = post_stage
        self.router = router
        self.tokenizer = Tokenizer()
        self.detokenizer = detokenizer
        self.residual_proj = residual_proj
        self.target_ratio = target_ratio
        self.inner_lr_ratio = inner_lr_ratio if inner_lr_ratio is not None else target_ratio
        self.detach_residual = detach_residual
        self.residual_drop_fn = residual_drop_fn
        self.ratio_override_fn = ratio_override_fn
        self.pad_parameter = pad_parameter

    def _get_effective_ratio(self, train_step: tuple[int, int] | None) -> float:
        """Return the effective target ratio, respecting ratio_override_fn if set."""
        if self.ratio_override_fn is not None and train_step is not None:
            iter_num, max_iters = train_step
            progress = iter_num / max_iters if max_iters > 0 else 1.0
            return self.ratio_override_fn(progress)
        return self.target_ratio

    def _ratio_loss(self, prob: Tensor, token_mask: Tensor, train_step: tuple[int, int] | None = None) -> Tensor:
        """Ratio loss to regularize compression toward target_ratio.

        ℒ = (N/(N-1)) * ((N-1)*F*G + (1-F)*(1-G))
        where F = fraction selected (discrete), G = mean prob (continuous), N = target ratio.
        Minimized when F = G = 1/N.
        """
        N = self._get_effective_ratio(train_step)
        F = token_mask.float().mean()
        G = prob.mean()
        return (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))

    def _get_residual_drop(self, train_step: tuple[int, int] | None) -> float:
        """Compute current residual drop rate."""
        if train_step is None:
            return 0.0  # no dropout at eval
        iter_num, max_iters = train_step
        progress = iter_num / max_iters if max_iters > 0 else 1.0
        return self.residual_drop_fn(progress)

    def forward(self, x: Tensor, state: dict, positions: Tensor | None = None, train_step: tuple[int, int] | None = None) -> tuple[Tensor, dict, Tensor, dict[str, float]]:
        B, L, D = x.shape

        # 1. Pre-stage (encoder)
        x, pre_state = self.pre_stage(x, state['pre'], positions=positions)

        # 2. Residual projection (in fp32 for stability)
        device_type = 'cuda' if x.is_cuda else ('mps' if x.is_mps else 'cpu')
        with torch.autocast(device_type=device_type, enabled=False):
            residual = self.residual_proj(x.float())

        # 3. Router — compute boundary probabilities
        prob, token_mask, router_state, router_metrics = self.router(x, state['router'])

        # 4. Ratio loss at this level
        aux_loss = self._ratio_loss(prob, token_mask, train_step=train_step)

        # 5. Tokenize — compact to boundary positions
        chunked, counts, inner_positions = self.tokenizer(x, token_mask, positions=positions)

        # Track compression ratio: input_len / num_boundary_tokens
        n_tokens_selected = token_mask.float().sum(dim=1).mean().item()
        ratio = L / max(n_tokens_selected, 1.0)
        prob_mean = prob[:, 1:].mean().item() if L > 1 else 1.0

        # 6. Pad to deeper dimension if needed
        B_c, M, D_c = chunked.shape
        if self.pad_parameter is not None:
            pad = self.pad_parameter[None, None, :].expand(B_c, M, -1)
            chunked = torch.cat([chunked, pad], dim=-1)

        # 7. Main stage (on compressed sequence) — skip if empty
        metrics: dict[str, float] = {'ratio': ratio, 'prob_mean': prob_mean, **router_metrics}
        M = chunked.shape[1]
        if M > 0:
            if isinstance(self.main_stage, HNet):
                chunked, main_state, inner_aux, inner_metrics = self.main_stage(chunked, state['main'], positions=inner_positions, train_step=train_step)
                aux_loss = aux_loss + inner_aux
                for k, v in inner_metrics.items():
                    metrics[f'inner_{k}'] = v
            else:
                chunked, main_state = self.main_stage(chunked, state['main'], positions=inner_positions)
        else:
            main_state = state['main']

        # 8. Truncate back to this level's dimension
        chunked = chunked[..., :D]

        # 9. DeTokenize — EMA upsample + residual
        residual_drop = self._get_residual_drop(train_step)
        x, detok_state, detok_metrics = self.detokenizer(
            chunked, residual, token_mask, prob, counts, state['detokenizer'],
            detach_residual=self.detach_residual, residual_drop=residual_drop,
        )
        metrics['residual_drop'] = residual_drop
        metrics.update(detok_metrics)

        # 10. Post-stage (decoder)
        x, post_state = self.post_stage(x, state['post'], positions=positions)

        new_state = {
            'pre': pre_state,
            'main': main_state,
            'post': post_state,
            'router': router_state,
            'detokenizer': detok_state,
        }
        return x, new_state, aux_loss, metrics

    def optim_groups(self, lr_scale: float = 1.0) -> list[dict]:
        from utils import decay_nodecay_groups
        outer_params = list(self.pre_stage.parameters()) + \
                       list(self.post_stage.parameters()) + \
                       list(self.router.parameters()) + \
                       list(self.residual_proj.parameters()) + \
                       list(self.detokenizer.parameters())
        if self.pad_parameter is not None:
            outer_params.append(self.pad_parameter)
        groups = decay_nodecay_groups(outer_params, lr_scale)

        inner_scale = lr_scale / self.inner_lr_ratio
        if isinstance(self.main_stage, HNet):
            groups += self.main_stage.optim_groups(lr_scale=inner_scale)
        else:
            groups += decay_nodecay_groups(self.main_stage.parameters(), inner_scale)
        return groups

    def initial_state(self, batch_size):
        return {
            'pre': self.pre_stage.initial_state(batch_size),
            'main': self.main_stage.initial_state(batch_size),
            'post': self.post_stage.initial_state(batch_size),
            'router': self.router.initial_state(batch_size),
            'detokenizer': self.detokenizer.initial_state(batch_size),
        }

    def flops_per_fwdbwd(self):
        return self.pre_stage.flops_per_fwdbwd() + \
               self.main_stage.flops_per_fwdbwd() + \
               self.post_stage.flops_per_fwdbwd()


class HNetLM(nn.Module):
    """H-Net Language Model wrapper. Conforms to TrainModel protocol.

    Wraps a fully-assembled HNet backbone with token embeddings, positional
    embeddings, and an LM head.
    """

    def __init__(
        self,
        backbone: HNet,
        embeddings: nn.Embedding,
        lm_head: nn.Linear,
        wpe: nn.Embedding | None = None,
        ratio_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.backbone = backbone
        self.embeddings = embeddings
        self.lm_head = lm_head
        self.wpe = wpe
        self.ratio_loss_weight = ratio_loss_weight

    def forward(self, idx: Tensor, state, targets: Tensor | None = None, train_step: tuple[int, int] | None = None):
        """TrainModel protocol: (logits, state, loss, metrics)."""
        L = idx.size(1)
        offset = state['pre'][0][0].size(2)  # cached seq len
        positions = torch.arange(offset, offset + L, dtype=torch.long, device=idx.device)
        x = self.embeddings(idx)
        if self.wpe is not None:
            x = x + self.wpe(positions)
        x, state, ratio_loss, metrics = self.backbone(x, state, positions=positions, train_step=train_step)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            metrics['ce_loss'] = loss.item()
            metrics['ratio_loss'] = ratio_loss.item()
            loss = loss + self.ratio_loss_weight * ratio_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, state, loss, metrics

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def optim_groups(self) -> list[dict]:
        from utils import decay_nodecay_groups
        embed_params = list(self.embeddings.parameters()) + \
                       list(self.lm_head.parameters())
        if self.wpe is not None:
            embed_params += list(self.wpe.parameters())
        groups = decay_nodecay_groups(embed_params)
        groups += self.backbone.optim_groups()
        return groups

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        # Feed prompt or single token, sample, repeat
        cur = idx
        for _ in range(max_new_tokens):
            logits, state, _, _ = self(cur, state)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            cur = idx_next  # subsequent iterations feed only the new token
        return idx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not getattr(module.weight, '_no_reinit', False):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()


class StageLM(nn.Module):
    """Flat (no-hierarchy) LM wrapper. Conforms to TrainModel protocol.

    Wraps a bare Stage with token embeddings, positional embeddings, and LM head.
    Used when the shape string is a leaf (e.g. "6x64") with no HNet hierarchy.
    """

    def __init__(
        self,
        backbone: Stage,
        embeddings: nn.Embedding,
        lm_head: nn.Linear,
        wpe: nn.Embedding | None = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.embeddings = embeddings
        self.lm_head = lm_head
        self.wpe = wpe

    def forward(self, idx: Tensor, state, targets: Tensor | None = None, train_step: tuple[int, int] | None = None):
        L = idx.size(1)
        offset = state[0][0].size(2)  # cached seq len from first block's K cache
        positions = torch.arange(offset, offset + L, dtype=torch.long, device=idx.device)
        x = self.embeddings(idx)
        if self.wpe is not None:
            x = x + self.wpe(positions)
        x, state = self.backbone(x, state, positions=positions)

        metrics: dict[str, float] = {}
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            metrics['ce_loss'] = loss.item()
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, state, loss, metrics

    def initial_state(self, batch_size):
        return self.backbone.initial_state(batch_size)

    def optim_groups(self) -> list[dict]:
        from utils import decay_nodecay_groups
        params = list(self.parameters())
        return decay_nodecay_groups(params)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        cur = idx
        for _ in range(max_new_tokens):
            logits, state, _, _ = self(cur, state)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            cur = idx_next
        return idx

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not getattr(module.weight, '_no_reinit', False):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0

    def flops_per_fwdbwd(self):
        return self.backbone.flops_per_fwdbwd()

