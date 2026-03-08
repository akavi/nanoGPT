import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os

from models.hnet.model import HNet, HNetLM, StageLM, Stage, DeTokenizer
from models.hnet.router import Router, CosineSimRouting, LinearSigmoidRouting, MLPSigmoidRouting
from models.model import CsaConfig, CsaBlock
from train import train, TrainConfig
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_fixed_batch,
    get_sampled_batch,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face

overridable = override(sys.argv, {
    "out_dir": "out-face-hnet",
    "dataset": "image_anime_face",
    "mode": "from_scratch",
    "device": "cuda",
    "seed": 1337,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "bias": False,
    "block_size": 1024,
    "batch_size": 128,
    "max_iters": 3000,
    "sampled_batch": True,
    # H-Net options
    "shape": "balanced",       # "balanced" or "narrow_shell"
    "raster": "linear",        # "linear" or "hilbert"
    "ratio_loss_weight": 0.03,
    "inner_lr_ratio": 0.0,         # 0 = use target_ratio; set to override inner stage LR scaling
    "detach_residual": False,       # detach residual path so loss gradients flow only through main stage
    "residual_drop": 0.0,          # prob of dropping residual path during training (stochastic depth)
    "routing": "cosine_sim",       # "cosine_sim", "linear_sigmoid", or "mlp_sigmoid"
    "pos_emb": "rope_2d",          # "learned", "rope_1d", or "rope_2d"
})

# -----------------------------------------------------------------------------#
# Shape string parser
# -----------------------------------------------------------------------------#
# Syntax:  Shape = NxD | NxDxH | NxD-R(Shape)-NxD | NxDxH-R(Shape)-NxDxH
# Examples:
#   "2x288-4(5x480)-2x288"                — 1-level HNet
#   "2x288-4(3x480-4(5x640)-3x480)-2x288" — 2-level nested HNet
#   "5x384"                                — flat (no HNet, just a Stage)
#
# Named aliases for backwards compat:

SHAPE_ALIASES = {
    "balanced":     "2x288-4(5x480)-2x288",
    "narrow_shell": "1x128-4(10x384)-1x128",
    "isotropic":    "2x384-4(6x384)-2x384",
}

shape = overridable['shape']
shape_str = SHAPE_ALIASES.get(shape, shape)

# -----------------------------------------------------------------------------#
# Rasterization order
# -----------------------------------------------------------------------------#

BOS_ID = 0
H, W, C = 32, 32, 1
TOKENS_LINEAR = H * W * C

def _hilbert_d2xy(n, d):
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if ((d & 1) ^ rx) else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y

def _build_hilbert_order(h, w):
    n = max(h, w)
    assert n & (n - 1) == 0, "Hilbert curve requires power-of-2 grid"
    order = []
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if x < w and y < h:
            order.append(y * w + x)
    return order

raster = overridable['raster']
assert raster in ("linear", "hilbert"), f"raster must be 'linear' or 'hilbert', got {raster!r}"

if raster == "hilbert":
    HILBERT_ORDER = _build_hilbert_order(H, W)
    INV_HILBERT_ORDER = [0] * len(HILBERT_ORDER)
    for i, li in enumerate(HILBERT_ORDER):
        INV_HILBERT_ORDER[li] = i
    HILBERT_ORDER_T = torch.tensor(HILBERT_ORDER, dtype=torch.long)
    INV_HILBERT_ORDER_T = torch.tensor(INV_HILBERT_ORDER, dtype=torch.long)

# Build pos_coords for rope_2d: map each sequence position to (x, y) on the grid
pos_coords = None
if overridable['pos_emb'] == 'rope_2d':
    if raster == 'hilbert':
        coords_list = [_hilbert_d2xy(max(H, W), d) for d in range(H * W)]
    else:
        coords_list = [(i % W, i // W) for i in range(H * W)]
    # Pad position 0 (BOS) with (0, 0)
    pos_coords = torch.tensor([(0, 0)] + coords_list, dtype=torch.long, device=overridable['device'])


# -----------------------------------------------------------------------------#
# Factory
# -----------------------------------------------------------------------------#

def _make_attn_stage(n_blocks, d_model, n_head, block_size, bias, dropout, rope_fn=None):
    """Build a Stage of CsaBlock attention layers."""
    csa_cfg = CsaConfig(
        n_head=n_head, n_embd=d_model, n_step=1,
        block_size=block_size, bias=bias, dropout=dropout,
    )
    blocks = [CsaBlock(csa_cfg, i, rope_fn=rope_fn) for i in range(n_blocks)]
    return Stage(blocks, d_model)


def _make_residual_proj(d_model):
    proj = torch.nn.Linear(d_model, d_model, bias=True, dtype=torch.float32)
    torch.nn.init.normal_(proj.weight, std=0.01)
    torch.nn.init.zeros_(proj.bias)
    proj.weight._no_reinit = True
    proj.bias._no_reinit = True
    return proj


import re

def _parse_cluster(s):
    """Parse 'NxD' or 'NxDxH' into (n_layers, d_model, n_head_or_None)."""
    parts = s.split('x')
    assert len(parts) in (2, 3), f"Bad cluster spec: {s!r}, expected NxD or NxDxH"
    n, d = int(parts[0]), int(parts[1])
    h = int(parts[2]) if len(parts) == 3 else None
    return n, d, h

def parse_shape(s):
    """Parse shape string recursively.

    Returns one of:
      ('leaf', n_layers, d_model, n_head)           — a single Stage
      ('hnet', pre, ratio, inner, post)              — an HNet level
        where pre/post = (n_layers, d_model, n_head)
        and inner is another parsed shape (recursive)
    """
    # Try to match: cluster-R(inner)-cluster
    # Find the first R( pattern
    m = re.match(r'^(\d+x\d+(?:x\d+)?)-(\d+)\((.+)\)-(\d+x\d+(?:x\d+)?)$', s)
    if m:
        pre_str, ratio_str, inner_str, post_str = m.group(1), m.group(2), m.group(3), m.group(4)
        # But the inner_str might contain nested parens, so we need to find the matching paren
        # Re-parse properly with paren matching
        pass

    # Paren-aware parsing
    # Find first R( where R is a number
    idx = s.find('(')
    if idx == -1:
        # Leaf: just NxD or NxDxH
        n, d, h = _parse_cluster(s)
        return ('leaf', n, d, h)

    # Find the ratio number before '('
    dash_before = s.rfind('-', 0, idx)
    assert dash_before != -1, f"Expected dash before ratio in: {s!r}"
    pre_str = s[:dash_before]
    ratio_str = s[dash_before+1:idx]
    ratio = int(ratio_str)

    # Find matching closing paren
    depth = 1
    i = idx + 1
    while i < len(s) and depth > 0:
        if s[i] == '(':
            depth += 1
        elif s[i] == ')':
            depth -= 1
        i += 1
    assert depth == 0, f"Unmatched parens in: {s!r}"
    inner_str = s[idx+1:i-1]
    post_str = s[i+1:]  # skip the '-' after ')'
    assert post_str, f"Missing post cluster in: {s!r}"

    pre_n, pre_d, pre_h = _parse_cluster(pre_str)
    post_n, post_d, post_h = _parse_cluster(post_str)
    assert pre_d == post_d, f"Pre/post dim mismatch: {pre_d} vs {post_d}"

    inner = parse_shape(inner_str)
    return ('hnet', (pre_n, pre_d, pre_h), ratio, inner, (post_n, post_d, post_h))



def _make_router(routing: str, d_model: int) -> Router:
    """Build a Router with the specified strategy."""
    strategies = {
        'cosine_sim': lambda: CosineSimRouting(d_model),
        'linear_sigmoid': lambda: LinearSigmoidRouting(d_model),
        'mlp_sigmoid': lambda: MLPSigmoidRouting(d_model),
    }
    if routing not in strategies:
        raise ValueError(f"Unknown routing strategy {routing!r}, expected one of {list(strategies)}")
    return Router(d_model, strategies[routing]())


def _build(parsed, block_size, n_head_default, bias, dropout, rope_fn, routing='cosine_sim', inner_lr_ratio=None, detach_residual=False, residual_drop=0.0):
    """Recursively build from a parsed shape. Returns (module, d_model).

    For 'leaf': returns (Stage, d_model)
    For 'hnet': returns (HNet, d_model)
    """
    if parsed[0] == 'leaf':
        _, n_layers, d_model, n_head = parsed
        n_head = n_head or n_head_default
        stage = _make_attn_stage(n_layers, d_model, n_head, block_size, bias, dropout, rope_fn=rope_fn)
        return stage, d_model

    _, (pre_n, pre_d, pre_h), ratio, inner_parsed, (post_n, post_d, post_h) = parsed
    d_model = pre_d
    pre_h = pre_h or n_head_default
    post_h = post_h or n_head_default

    inner_block_size = max(1, int(block_size / ratio))

    pre_stage = _make_attn_stage(pre_n, d_model, pre_h, block_size, bias, dropout, rope_fn=rope_fn)
    post_stage = _make_attn_stage(post_n, d_model, post_h, block_size, bias, dropout, rope_fn=rope_fn)
    main_stage, d_inner = _build(inner_parsed, inner_block_size, n_head_default, bias, dropout, rope_fn, routing=routing, inner_lr_ratio=inner_lr_ratio, detach_residual=detach_residual, residual_drop=residual_drop)

    pad_dim = d_inner - d_model
    hnet = HNet(
        d_model=d_model,
        pre_stage=pre_stage,
        main_stage=main_stage,
        post_stage=post_stage,
        router=_make_router(routing, d_model),
        detokenizer=DeTokenizer(d_model),
        residual_proj=_make_residual_proj(d_model),
        pad_parameter=torch.nn.Parameter(torch.zeros(pad_dim)) if pad_dim > 0 else None,
        target_ratio=float(ratio),
        inner_lr_ratio=inner_lr_ratio,
        detach_residual=detach_residual,
        residual_drop=residual_drop,
    )
    return hnet, d_model


def make_hnet(
    shape_str,
    block_size,
    vocab_size,
    n_head=8,
    bias=False,
    dropout=0.0,
    ratio_loss_weight=0.01,
    pos_emb='learned',
    pos_coords=None,
    routing='cosine_sim',
    inner_lr_ratio=None,
    detach_residual=False,
    residual_drop=0.0,
):
    nn = torch.nn
    parsed = parse_shape(shape_str)

    # Build rope_fn closure if using RoPE
    rope_fn = None
    if pos_emb == 'rope_1d':
        from models.model import apply_rope_1d
        rope_fn = apply_rope_1d
    elif pos_emb == 'rope_2d':
        from models.model import apply_rope_2d
        assert pos_coords is not None, "pos_coords required for rope_2d"
        coords = pos_coords
        rope_fn = lambda q, k, positions: apply_rope_2d(q, k, positions, coords)

    backbone, d_model = _build(parsed, block_size, n_head, bias, dropout, rope_fn, routing=routing, inner_lr_ratio=inner_lr_ratio, detach_residual=detach_residual, residual_drop=residual_drop)

    embeddings = nn.Embedding(vocab_size, d_model)
    lm_head = nn.Linear(d_model, vocab_size, bias=False)
    wpe = nn.Embedding(block_size, d_model) if pos_emb == 'learned' else None

    if isinstance(backbone, HNet):
        model = HNetLM(
            backbone=backbone,
            embeddings=embeddings,
            lm_head=lm_head,
            wpe=wpe,
            ratio_loss_weight=ratio_loss_weight,
        )
    else:
        model = StageLM(
            backbone=backbone,
            embeddings=embeddings,
            lm_head=lm_head,
            wpe=wpe,
        )

    model.apply(model._init_weights)
    print(f"number of parameters: {model.get_num_params()/1e6:.2f}M")
    return model


# -----------------------------------------------------------------------------#
# Init Model
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_sampled_data(overridable['dataset'], prepare_image_anime_face)

model = make_hnet(
    shape_str=shape_str,
    vocab_size=256,
    block_size=overridable['block_size'],
    bias=overridable['bias'],
    dropout=0.0,
    ratio_loss_weight=overridable['ratio_loss_weight'],
    pos_emb=overridable['pos_emb'],
    pos_coords=pos_coords,
    routing=overridable['routing'],
    inner_lr_ratio=overridable['inner_lr_ratio'] or None,
    detach_residual=overridable['detach_residual'],
    residual_drop=overridable['residual_drop'],
)

optimizer_config = OptimizerConfig(
    weight_decay=1e-1,
    learning_rate=overridable['learning_rate'],
    betas=(0.9, 0.95),
    device=overridable['device'],
)

mode = overridable['mode']
if mode == "from_scratch":
    iter_num = 0
    best_val_loss = 1e9
    optimizer = configure_optimizers(model, optimizer_config)
elif mode == "resume" or mode == "sample":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        overridable['out_dir'], overridable['device'], model, optimizer_config,
    )
    print("Best val loss: ", best_val_loss)
else:
    raise ValueError(f"Unsupported mode={mode}")

# -----------------------------------------------------------------------------#
# Data utils
# -----------------------------------------------------------------------------#

def detokenize(tokens: torch.Tensor) -> Image.Image:
    t = tokens[1:].cpu()
    if raster == "hilbert":
        linear = torch.zeros(TOKENS_LINEAR, dtype=t.dtype)
        linear[HILBERT_ORDER_T] = t
        t = linear
    img = np.asarray(t, dtype=np.uint8).reshape(H, W)
    return Image.fromarray(img, mode="L")

_get_raw_batch = get_sampled_batch if overridable['sampled_batch'] else get_fixed_batch

def get_batch(split, batch_size):
    rows = _get_raw_batch(
        split,
        batch_size,
        DataConfig(
            dataset=overridable['dataset'],
            device=overridable['device'],
        ),
    )
    if raster == "hilbert":
        rows = rows[:, HILBERT_ORDER_T.to(rows.device)]

    block_size = overridable['block_size']
    first_col = torch.full((batch_size, 1), BOS_ID, dtype=rows.dtype, device=rows.device)
    x_out = torch.cat([first_col, rows[:, :-1]], dim=1).long()
    y_out = rows[:, 0:block_size + 1].contiguous().long()
    return x_out, y_out

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

train_config = TrainConfig(
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    device=overridable['device'],

    decay_lr=True,
    lr_decay_iters=3000,
    learning_rate=overridable['learning_rate'],
    min_lr=overridable['min_lr'],

    grad_clip=1.0,
    gradient_accumulation_steps=1,
    batch_size=overridable['batch_size'],

    eval_only=False,
    eval_interval=250,
    eval_iters=20,
    warmup_iters=300,
    max_iters=overridable['max_iters'],

    log_interval=1,
    always_save_checkpoint=True,
    out_dir=overridable['out_dir'],
    compile=False,
)

if mode == "resume" or mode == "from_scratch":
    train(
        model=model,
        optimizer=optimizer,
        get_batch=get_batch,
        save_checkpoint=lambda it, val_loss, cfg, mdl, opt: save_config_checkpoint(
            overridable['out_dir'], it, val_loss, cfg, mdl, opt,
        ),
        config=train_config,
    )
else:
    def init_gen(device):
        return torch.zeros((1, 1), dtype=int, device=device)

    def detokenize_and_save(tokens: np.ndarray, path: str):
        img = detokenize(tokens)
        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=1024,
        temperature=0.8,
        top_k=200,
        seed=1337,
        device=overridable['device'],
        compile=False,
    )
    os.makedirs(overridable['out_dir'], exist_ok=True)
    sample(
        model=model,
        init_gen=init_gen,
        detokenize=detokenize_and_save,
        config=sample_config,
    )
