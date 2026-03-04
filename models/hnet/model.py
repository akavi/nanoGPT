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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.mamba import RMSNorm


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
# CosineSimRouter
# ---------------------------------------------------------------------------

class CosineSimRouter(nn.Module):
    """Computes boundary probabilities from adjacent cosine similarities.

    Position 0 always gets prob=1.0 (always a boundary).
    For position i>0: prob = (1 - cos_sim(q(h[i-1]), k(h[i]))) / 2.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        # Initialize as identity so routing starts neutral
        nn.init.eye_(self.q_proj.weight)
        nn.init.eye_(self.k_proj.weight)
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def prob_boundary(self, hidden_states: Tensor) -> Tensor:
        """Compute per-position boundary probabilities.

        Args:
            hidden_states: (B, L, D)
        Returns:
            (B, L) boundary probabilities in [0, 1].
        """
        B, L, D = hidden_states.shape
        if L <= 1:
            return torch.ones(B, L, device=hidden_states.device, dtype=torch.float32)

        cos_sim = torch.einsum(
            'b l d, b l d -> b l',
            F.normalize(self.q_proj(hidden_states[:, :-1]).float(), dim=-1),
            F.normalize(self.k_proj(hidden_states[:, 1:]).float(), dim=-1),
        )  # (B, L-1)

        default = torch.ones(B, 1, device=hidden_states.device, dtype=cos_sim.dtype)
        prob = torch.cat([default, ((1 - cos_sim) / 2).clamp(0, 1)], dim=1)
        return prob

    def forward(
        self,
        hidden_states: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Compute routing decisions.

        Args:
            hidden_states: (B, L, D)
            state: (last_token (B,D), has_seen_token (B,)) or None

        Returns:
            (prob, token_mask, new_state)
            - prob: (B, L) boundary probabilities
            - token_mask: (B, L) boolean — True where prob > 0.5
            - new_state: (last_token, has_seen_token)
        """
        B, L, D = hidden_states.shape

        prob = self.prob_boundary(hidden_states)  # (B, L)

        # Handle cross-boundary state from previous forward call
        if state is not None:
            last_token, has_seen_token = state
            # Compute boundary prob between last_token and first token of each sequence
            pairs = torch.stack([last_token, hidden_states[:, 0]], dim=1)  # (B, 2, D)
            cross_prob = self.prob_boundary(pairs)[:, 1]  # (B,)
            prob = prob.clone()
            prob[:, 0] = torch.where(has_seen_token, cross_prob, torch.ones_like(cross_prob))
        else:
            prob = prob.clone()
            prob[:, 0] = 1.0  # First position is always a boundary

        token_mask = prob > 0.5

        # Update state
        new_state = (
            hidden_states[:, -1].detach().clone(),  # last_token
            torch.ones(B, device=hidden_states.device, dtype=torch.bool),  # has_seen_token
        )

        return prob, token_mask, new_state

    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.d_model, device=device),
            torch.zeros(batch_size, device=device, dtype=torch.bool),
        )


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

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('_device_buf', torch.empty(0), persistent=False)

    def forward(
        self,
        hidden_states: Tensor,  # (B, M, D) — compacted chunk states
        residual: Tensor,       # (B, L, D) — pre-router residual
        token_mask: Tensor,     # (B, L) bool
        prob: Tensor,           # (B, L) boundary probabilities
        counts: Tensor,         # (B,) tokens per batch element
        state: Tensor | None = None,  # (B, D) previous EMA value
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            output: (B, L, D)
            new_state: (B, D) last EMA value per batch element
        """
        B, L, D = residual.shape
        M = hidden_states.shape[1]
        device = hidden_states.device

        if M == 0:
            # No boundary tokens this step — use previous EMA state to match
            # training behavior where non-boundary positions still receive the
            # EMA-broadcast main-stage contribution via chunk_idx mapping.
            contribution = state.unsqueeze(1).expand(B, L, D)
            output = (residual.float() + contribution.float()).to(residual.dtype)
            return output, state

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
        output = (residual.float() + long_states.float()).to(out_dtype)

        # Compute new state: last EMA output per batch element
        new_state = residual.new_zeros(B, D)
        for b in range(B):
            c = int(counts[b].item())
            if c > 0:
                new_state[b] = ema_out[b, c - 1]
            elif state is not None:
                new_state[b] = state[b]

        return output, new_state

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
        router: CosineSimRouter,
        detokenizer: DeTokenizer,
        residual_proj: nn.Linear,
        pad_parameter: nn.Parameter | None,
        target_ratio: float = 4.0,
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
        self.pad_parameter = pad_parameter

    def _ratio_loss(self, prob: Tensor, token_mask: Tensor) -> Tensor:
        """Ratio loss to regularize compression toward target_ratio.

        ℒ = (N/(N-1)) * ((N-1)*F*G + (1-F)*(1-G))
        where F = fraction selected (discrete), G = mean prob (continuous), N = target ratio.
        Minimized when F = G = 1/N.
        """
        N = self.target_ratio
        F = token_mask.float().mean()
        G = prob.mean()
        return (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))

    def forward(self, x: Tensor, state: dict, positions: Tensor | None = None) -> tuple[Tensor, dict, Tensor, dict[str, float]]:
        B, L, D = x.shape

        # 1. Pre-stage (encoder)
        x, pre_state = self.pre_stage(x, state['pre'], positions=positions)

        # 2. Residual projection (in fp32 for stability)
        device_type = 'cuda' if x.is_cuda else ('mps' if x.is_mps else 'cpu')
        with torch.autocast(device_type=device_type, enabled=False):
            residual = self.residual_proj(x.float())

        # 3. Router — compute boundary probabilities
        prob, token_mask, router_state = self.router(x, state['router'])

        # 4. Ratio loss at this level
        aux_loss = self._ratio_loss(prob, token_mask)

        # 5. Tokenize — compact to boundary positions
        chunked, counts, inner_positions = self.tokenizer(x, token_mask, positions=positions)

        # Track compression ratio: input_len / num_boundary_tokens
        n_tokens_selected = token_mask.float().sum(dim=1).mean().item()
        ratio = L / max(n_tokens_selected, 1.0)

        # 6. Pad to deeper dimension if needed
        B_c, M, D_c = chunked.shape
        if self.pad_parameter is not None:
            pad = self.pad_parameter[None, None, :].expand(B_c, M, -1)
            chunked = torch.cat([chunked, pad], dim=-1)

        # 7. Main stage (on compressed sequence) — skip if empty
        metrics: dict[str, float] = {'ratio': ratio}
        M = chunked.shape[1]
        if M > 0:
            if isinstance(self.main_stage, HNet):
                chunked, main_state, inner_aux, inner_metrics = self.main_stage(chunked, state['main'], positions=inner_positions)
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
        x, detok_state = self.detokenizer(
            chunked, residual, token_mask, prob, counts, state['detokenizer'],
        )

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

        inner_scale = lr_scale / self.target_ratio
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

    def forward(self, idx: Tensor, state, targets: Tensor | None = None):
        """TrainModel protocol: (logits, state, loss, metrics)."""
        L = idx.size(1)
        offset = state['pre'][0][0].size(2)  # cached seq len
        positions = torch.arange(offset, offset + L, dtype=torch.long, device=idx.device)
        x = self.embeddings(idx)
        if self.wpe is not None:
            x = x + self.wpe(positions)
        x, state, ratio_loss, metrics = self.backbone(x, state, positions=positions)

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

