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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.mamba import RMSNorm
from models.model import CsaConfig


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
    """Parallel-scan EMA: h[t] = decay[t] * h[t-1] + (1 - decay[t]) * x[t].

    Uses the associative scan identity:
        (a2, b2) ∘ (a1, b1) = (a2 * a1, a2 * b1 + b2)
    where a[t] = decay[t], b[t] = (1 - decay[t]) * x[t].

    All arithmetic stays in [0, 1] for the decay terms, so no overflow.
    """
    B, L, D = x.shape

    # a[t] = decay[t], b[t] = (1 - decay[t]) * x[t]
    a = decay.unsqueeze(-1).expand(B, L, D)       # (B, L, D)
    b = (1 - decay).unsqueeze(-1).expand(B, L, D) * x  # (B, L, D)

    # Fold initial state into the first position:
    # h[0] = decay[0] * init + (1 - decay[0]) * x[0]
    #       = a[0] * init + b[0]
    b = torch.cat([b[:, :1] + a[:, :1] * initial_state.unsqueeze(1), b[:, 1:]], dim=1)

    # Hillis-Steele parallel prefix scan (out-of-place for autograd)
    # Each step combines element i with element i-stride.
    # O(L log L) work, O(log L) depth — simple and correct.
    stride = 1
    while stride < L:
        b = torch.cat([b[:, :stride],
                        a[:, stride:] * b[:, :-stride] + b[:, stride:]], dim=1)
        a = torch.cat([a[:, :stride],
                        a[:, stride:] * a[:, :-stride]], dim=1)
        stride *= 2

    return b  # b now holds h[t] at each position


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

    def initial_state(self, batch_size, device='cpu'):
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
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            chunked: (B, M, D) where M = max tokens in any batch element
            counts: (B,) number of tokens per batch element
        """
        B, L, D = hidden_states.shape
        counts = token_mask.sum(dim=1)  # (B,)
        M = int(counts.max().item())

        if M == 0:
            return hidden_states.new_zeros(B, 0, D), counts

        chunked = hidden_states.new_zeros(B, M, D)
        for b in range(B):
            selected = hidden_states[b, token_mask[b]]  # (n_tokens, D)
            chunked[b, :selected.shape[0]] = selected

        return chunked, counts


class DeTokenizer(nn.Module):
    """EMA-based upsampling + STE weighting + residual add-back.

    Runs EMA over the compacted chunk states, then broadcasts back
    to original sequence length via cumulative chunk indexing.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

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

    def initial_state(self, batch_size, device='cpu'):
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

    def forward(self, x: Tensor, state: list) -> tuple[Tensor, list]:
        new_state = []
        for block, s in zip(self.blocks, state):
            x, s_new = block(x, s)
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
    """One level of the H-Net hierarchy.

    Contains:
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
        pad_parameter: nn.Parameter | None = None,
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
        if pad_parameter is not None:
            self.pad_parameter = pad_parameter
        else:
            self.pad_parameter = None

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

    def forward(self, x: Tensor, state: dict) -> tuple[Tensor, dict, Tensor, dict[str, float]]:
        B, L, D = x.shape

        # 1. Pre-stage (encoder)
        x, pre_state = self.pre_stage(x, state['pre'])

        # 2. Residual projection (in fp32 for stability)
        device_type = 'cuda' if x.is_cuda else ('mps' if x.is_mps else 'cpu')
        with torch.autocast(device_type=device_type, enabled=False):
            residual = self.residual_proj(x.float())

        # 3. Router — compute boundary probabilities
        prob, token_mask, router_state = self.router(x, state['router'])

        # 4. Ratio loss at this level
        aux_loss = self._ratio_loss(prob, token_mask)

        # 5. Tokenize — compact to boundary positions
        chunked, counts = self.tokenizer(x, token_mask)

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
                chunked, main_state, inner_aux, inner_metrics = self.main_stage(chunked, state['main'])
                aux_loss = aux_loss + inner_aux
                # prefix inner metrics to distinguish levels
                for k, v in inner_metrics.items():
                    metrics[f'inner_{k}'] = v
            else:
                chunked, main_state = self.main_stage(chunked, state['main'])
        else:
            main_state = state['main']

        # 8. Truncate back to this level's dimension
        chunked = chunked[..., :D]

        # 9. DeTokenize — EMA upsample + residual
        x, detok_state = self.detokenizer(
            chunked, residual, token_mask, prob, counts, state['detokenizer'],
        )

        # 10. Post-stage (decoder)
        x, post_state = self.post_stage(x, state['post'])

        new_state = {
            'pre': pre_state,
            'main': main_state,
            'post': post_state,
            'router': router_state,
            'detokenizer': detok_state,
        }
        return x, new_state, aux_loss, metrics

    def initial_state(self, batch_size, device='cpu'):
        return {
            'pre': self.pre_stage.initial_state(batch_size),
            'main': self.main_stage.initial_state(batch_size, device=device) if isinstance(self.main_stage, HNet) else self.main_stage.initial_state(batch_size),
            'post': self.post_stage.initial_state(batch_size),
            'router': self.router.initial_state(batch_size, device=device),
            'detokenizer': self.detokenizer.initial_state(batch_size, device=device),
        }


# ---------------------------------------------------------------------------
# HNetLM — CausalLM wrapper conforming to TrainModel protocol
# ---------------------------------------------------------------------------

@dataclass
class HNetConfig:
    vocab_size: int = 256
    d_model: int = 128        # outer (shell) dimension
    d_inner: int = 384        # inner (core) dimension
    n_head: int = 8           # attention heads (used at both levels)
    n_pre: int = 1            # pre/post attention blocks (shell)
    n_post: int = 1
    n_main: int = 10          # main stage attention blocks (core)
    block_size: int = 1024    # max sequence length
    d_head_detok: int = 32
    detok_chunk_size: int = 256
    target_ratio: float = 4.0
    ratio_loss_weight: float = 0.01
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = False
    device: str = 'cuda'


def _make_attn_stage(n_blocks: int, d_model: int, cfg: HNetConfig, block_size: int) -> Stage:
    """Build a Stage of CsaBlock attention layers."""
    from models.model import CsaBlock
    csa_cfg = CsaConfig(
        n_head=cfg.n_head,
        n_embd=d_model,
        n_step=1,
        block_size=block_size,
        bias=cfg.bias,
        dropout=cfg.dropout,
    )
    # CsaBlock already does prenorm+residual, so use it directly (no Block wrapper)
    blocks = [CsaBlock(csa_cfg, i) for i in range(n_blocks)]
    return Stage(blocks, d_model)


def _make_residual_proj(d_model: int) -> nn.Linear:
    proj = nn.Linear(d_model, d_model, bias=True, dtype=torch.float32)
    nn.init.eye_(proj.weight)
    nn.init.zeros_(proj.bias)
    proj.weight._no_reinit = True
    proj.bias._no_reinit = True
    return proj


class HNetLM(nn.Module):
    """H-Net Language Model. Conforms to TrainModel protocol."""

    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

        self.backbone = self._build_backbone(config)

        self.apply(self._init_weights)
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}M")

    def _build_backbone(self, cfg: HNetConfig) -> HNet:
        # Estimate inner block_size as block_size / target_ratio
        inner_block_size = max(1, int(cfg.block_size / cfg.target_ratio))

        main_stage = _make_attn_stage(cfg.n_main, cfg.d_inner, cfg, inner_block_size)
        pre_stage = _make_attn_stage(cfg.n_pre, cfg.d_model, cfg, cfg.block_size)
        post_stage = _make_attn_stage(cfg.n_post, cfg.d_model, cfg, cfg.block_size)

        pad_dim = cfg.d_inner - cfg.d_model
        return HNet(
            d_model=cfg.d_model,
            pre_stage=pre_stage,
            main_stage=main_stage,
            post_stage=post_stage,
            router=CosineSimRouter(cfg.d_model),
            detokenizer=DeTokenizer(cfg.d_model),
            residual_proj=_make_residual_proj(cfg.d_model),
            pad_parameter=nn.Parameter(torch.zeros(pad_dim)) if pad_dim > 0 else None,
            target_ratio=cfg.target_ratio,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not getattr(module.weight, '_no_reinit', False):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None and not getattr(module.bias, '_no_reinit', False):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: Tensor, state, targets: Tensor | None = None):
        """TrainModel protocol: (logits, state, loss, metrics)."""
        x = self.embeddings(idx)  # (B, L, D)
        x, state, ratio_loss, metrics = self.backbone(x, state)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            metrics['ratio_loss'] = ratio_loss.item()
            loss = loss + self.config.ratio_loss_weight * ratio_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, state, loss, metrics

    def initial_state(self, batch_size):
        device = next(self.parameters()).device
        return self.backbone.initial_state(batch_size, device=device)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, state, temperature=1.0, top_k=None):
        # Process the prompt
        logits, state, _, _ = self(idx, state)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            # Feed only the new token, relying on state
            logits, state, _, _ = self(idx_next, state)
        return idx

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        return -1.0

    def flops_per_fwdbwd(self):
        return self.backbone.pre_stage.flops_per_fwdbwd() + \
               self.backbone.main_stage.flops_per_fwdbwd() + \
               self.backbone.post_stage.flops_per_fwdbwd()
