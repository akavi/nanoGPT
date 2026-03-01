"""Python config for constructing a 2-stage XL HNet model.

Usage:
    from configs.model.hnet_2stage_XL import make_hnet_2stage_xl
    model = make_hnet_2stage_xl()
"""

from __future__ import annotations

from functools import partial
from itertools import chain
from typing import Any

import torch
import torch.nn as nn

from src.models.sequence.attention.sa import MultiheadSelfAttention
from src.models.sequence.hnet.block import Block
from src.models.sequence.hnet.hnet import (
    DynamicTokenizerModel,
    DynamicTokenizerModelWrapper,
    init_weights as init_hnet_weights,
    parse_layer_layout,
)
from src.models.sequence.hnet.router import CosineSimRouting, Router
from src.models.sequence.hnet.stage import Stage
from src.models.sequence.hnet.tokenizer import DeTokenizer, Tokenizer
from src.models.sequence.modules.ffn import FFN
from src.models.sequence.modules.ssd import SSD

# ---------------------------------------------------------------------------
# Shared layer config dicts
# ---------------------------------------------------------------------------

SSD_BASE = dict(
    d_state=128,
    d_conv=4,
    expand=2,
    chunk_size=256,
    norm_before_gate=False,
    conv_bias=True,
    dt_tied=True,
    A_init_range=(1, 16),
)

SA_BASE = dict(
    bias=False,
    use_rope=True,
    kv_heads=None,
    rope_kwargs=dict(
        rope_type="default",
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
    ),
)

FFN_BASE = dict(expand=None, activation="swish", glu=True, bias=False)


# ---------------------------------------------------------------------------
# Layer factory helpers (return callables that produce fresh modules)
# ---------------------------------------------------------------------------

def _ssd(d_model, **kw):
    return lambda: SSD(
            d_model, **SSD_BASE, **kw)


def _ffn(d_model, **kw):
    return lambda: FFN(d_model, **FFN_BASE, **kw)


def _sa(d_model, **kw):
    return lambda: MultiheadSelfAttention(d_model, **SA_BASE, **kw)


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------

def make_stage(
    layout: str,
    layer_factories: dict[str, Any],
    d_model: int,
    residual_in_fp32: bool = True,
    track_flops: bool = False,
) -> Stage:
    """Build a Stage from a layout DSL string and factory callables."""
    names = parse_layer_layout(layout)
    blocks = [Block(layer_factories[name](), d_model, residual_in_fp32) for name in names]
    return Stage(blocks, d_model, track_flops)


# ---------------------------------------------------------------------------
# Residual projection helper
# ---------------------------------------------------------------------------

def _make_residual_proj(
    d_model: int,
    init_mode: str = "identity",
    device=None,
    dtype=None,
) -> nn.Linear:
    proj = nn.Linear(d_model, d_model, bias=True, device=device, dtype=torch.float32)
    if init_mode == "identity":
        nn.init.eye_(proj.weight)
    elif init_mode == "zero":
        nn.init.zeros_(proj.weight)
    else:
        raise ValueError(f"Invalid init_mode: {init_mode}")
    nn.init.zeros_(proj.bias)
    proj.weight._no_reinit = True
    return proj


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(model: DynamicTokenizerModelWrapper, initializer_range: float):
    """Initialize all weights following the original hnet scheme."""
    _init_backbone(model.backbone, initializer_range, parent_residuals=0)


def _init_backbone(
    dtm: DynamicTokenizerModel,
    initializer_range: float,
    parent_residuals: int,
):
    n_res = parent_residuals + dtm.pre_stage.n_residual() + dtm.post_stage.n_residual()

    for stage in [dtm.pre_stage, dtm.post_stage]:
        stage.apply(partial(init_hnet_weights, n_layer=n_res, initializer_range=initializer_range))

    if isinstance(dtm.main_stage, Stage):
        dtm.main_stage.apply(
            partial(
                init_hnet_weights,
                n_layer=n_res + dtm.main_stage.n_residual(),
                initializer_range=initializer_range,
            )
        )
    else:
        _init_backbone(dtm.main_stage, initializer_range, n_res)


# ---------------------------------------------------------------------------
# LR multiplier application
# ---------------------------------------------------------------------------

def _set_lr(p: nn.Parameter, multiplier: float):
    optim = getattr(p, "_optim", {})
    optim["lr_multiplier"] = multiplier
    setattr(p, "_optim", optim)


def apply_lr_multipliers(
    model: DynamicTokenizerModelWrapper,
    multipliers: list[float],
):
    _apply_lr_backbone(model.backbone, multipliers, depth=0)
    for p in chain(model.embeddings.parameters(), model.lm_head.parameters()):
        _set_lr(p, multipliers[0])


def _apply_lr_backbone(
    dtm: DynamicTokenizerModel,
    multipliers: list[float],
    depth: int,
):
    for mod in [dtm.pre_stage, dtm.router, dtm.tokenizer, dtm.detokenizer, dtm.residual_proj, dtm.post_stage]:
        for p in mod.parameters():
            _set_lr(p, multipliers[depth])

    if dtm.pad_parameter is not None:
        _set_lr(dtm.pad_parameter, multipliers[depth + 1])

    if isinstance(dtm.main_stage, Stage):
        for p in dtm.main_stage.parameters():
            _set_lr(p, multipliers[depth + 1])
    else:
        _apply_lr_backbone(dtm.main_stage, multipliers, depth + 1)


# ---------------------------------------------------------------------------
# Full model construction
# ---------------------------------------------------------------------------

def make_hnet_2stage_xl(device=None, dtype=None) -> DynamicTokenizerModelWrapper:
    """Construct the full 2-stage XL HNet model."""
    fk = dict(device=device, dtype=dtype)

    # -- Model dimensions per depth -------------------------------------------
    D0 = 1024           # depth-0 (outer) model dim
    D1 = 1536           # depth-1 (inner) model dim
    D1_FFN = 4096       # depth-1 FFN hidden dim
    D2 = 2048            # depth-2 (core) model dim
    D2_FFN = 5504        # depth-2 FFN hidden dim

    # -- SA head sizes --------------------------------------------------------
    D_HEAD1 = 96        # depth-1 SA head dim
    D_HEAD2 = 128        # depth-2 SA head dim
    SA_MAX_CTX_D1 = 1023  # depth-1 SA max context length

    # -- Stage layouts --------------------------------------------------------
    LAYOUT_D0 = "ssd-4"
    LAYOUT_D1_PRE = "(sa-1-ffn-1)-1-ssd-4"
    LAYOUT_D1_POST = "ssd-4-(sa-1-ffn-1)-1"
    LAYOUT_D2 = "(sa-1-ffn-1)-27"

    # -- Tokeniser / CausalLM -------------------------------------------------
    VOCAB_SIZE = 256
    INITIALIZER_RANGE = 0.02
    LR_MULTIPLIERS = [3.0, 1.5, 0.75]

    # ----- Depth 0: pure SSD pre/post stages -----
    d0_factories = {"ssd": _ssd(D0, **fk)}
    pre_stage = make_stage(LAYOUT_D0, d0_factories, D0, track_flops=True)
    post_stage = make_stage(LAYOUT_D0, d0_factories, D0, track_flops=True)

    # ----- Depth 1: inner pre/post stages -----
    d1_factories = {
        "ssd": _ssd(D1, **fk),
        "sa": _sa(D1, d_head=D_HEAD1, max_context_len=SA_MAX_CTX_D1, **fk),
        "ffn": _ffn(D1, d_inner=D1_FFN),
    }
    inner_pre = make_stage(LAYOUT_D1_PRE, d1_factories, D1, track_flops=True)
    inner_post = make_stage(LAYOUT_D1_POST, d1_factories, D1, track_flops=True)

    # ----- Depth 2: innermost main stage -----
    d2_factories = {
        "sa": _sa(D2_FFN, d_head=D_HEAD2, **fk),
        "ffn": _ffn(D2_FFN, d_inner=D2_FFN),
    }
    inner_main = make_stage(LAYOUT_D2, d2_factories, D2, track_flops=True)

    # ----- Inner DynamicTokenizerModel (depth 1) -----
    inner_model = DynamicTokenizerModel(
        d_model=D1,
        pre_stage=inner_pre,
        main_stage=inner_main,
        post_stage=inner_post,
        router=Router(D1, CosineSimRouting(D1, init_mode="identity", **fk)),
        tokenizer=Tokenizer(),
        detokenizer=DeTokenizer(D1, residual_in_fp32=True, mode="v1"),
        residual_proj=_make_residual_proj(D1, "identity", **fk),
        pad_parameter=nn.Parameter(torch.zeros(D2 - D1)),
        residual_in_fp32=True,
        track_flops=True,
    )

    # ----- Outer DynamicTokenizerModel (depth 0) -----
    backbone = DynamicTokenizerModel(
        d_model=D0,
        pre_stage=pre_stage,
        main_stage=inner_model,
        post_stage=post_stage,
        router=Router(D0, CosineSimRouting(D0, init_mode="identity", **fk)),
        tokenizer=Tokenizer(),
        detokenizer=DeTokenizer(D0, residual_in_fp32=True, mode="v1"),
        residual_proj=_make_residual_proj(D0, "identity", **fk),
        pad_parameter=nn.Parameter(torch.zeros(D1 - D0)),
        residual_in_fp32=True,
        track_flops=True,
    )

    # ----- CausalLM wrapper -----
    model = DynamicTokenizerModelWrapper(
        backbone=backbone,
        vocab_size=VOCAB_SIZE,
        d_embed=D0,
        tie_embeddings=False,
    )

    # Weight init + LR multipliers
    init_weights(model, initializer_range=INITIALIZER_RANGE)
    apply_lr_multipliers(model, LR_MULTIPLIERS)

    return model


# ---------------------------------------------------------------------------
# State dict key conversion (original → DI-style)
# ---------------------------------------------------------------------------

def convert_state_dict(old_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert state dict keys from original DynamicTokenizerModelForCausalLM
    to DynamicTokenizerModelWrapper format.

    Key differences:
    - layers.{i}.layer. → blocks.{i}.layer.
    - layers.{i}.norm.norm. → blocks.{i}.norm.
    - <stage>.norm.norm. → <stage>.norm.
    """
    new_sd = {}
    for k, v in old_sd.items():
        new_k = k
        # Stage-level final norm: *.norm.norm.weight → *.norm.weight
        # Must be done before the layers→blocks replacement
        new_k = new_k.replace(".norm.norm.", ".norm.")
        # Block-level: layers.{i}.layer. → blocks.{i}.layer.
        # and layers.{i}.norm. → blocks.{i}.norm. (already handled by norm.norm → norm above)
        new_k = new_k.replace(".layers.", ".blocks.")
        new_sd[new_k] = v
    return new_sd


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Constructing model...")
    model = make_hnet_2stage_xl()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Forward pass smoke test
    B, L = 2, 128
    input_ids = torch.randint(0, 256, (B, L))
    print(f"Running forward pass with input shape ({B}, {L})...")
    with torch.no_grad():
        logits, state, metrics = model(input_ids)
    print(f"Output logits shape: {logits.shape}")
    print(f"FLOPs: {metrics.get(f'flops', 'N/A')}")
    print("Smoke test passed!")
