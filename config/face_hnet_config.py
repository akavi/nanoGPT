import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os

from models.hnet.model import HNetLM, HNetConfig
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
    "ratio_loss_weight": 0.01,
    "pos_emb": "learned",          # "learned", "rope_1d", or "rope_2d"
})

# -----------------------------------------------------------------------------#
# H-Net shape configs (param-matched to 10-layer CSA @ n_embd=384 ≈ 18.2M)
# -----------------------------------------------------------------------------#

SHAPES = {
    # D0=288, D1=480, 2/2 pre/post, 5 main — evenly distributed
    "balanced": HNetConfig(
        d_model=288,
        d_inner=480,
        n_head=8,
        n_pre=2,
        n_post=2,
        n_main=5,
        ratio_loss_weight=overridable['ratio_loss_weight'] ,
    ),
    # D0=128, D1=384, 1/1 pre/post, 10 main — thin encoder/decoder, deep core
    "narrow_shell": HNetConfig(
        d_model=128,
        d_inner=384,
        n_head=8,
        n_pre=1,
        n_post=1,
        n_main=10,
        ratio_loss_weight=overridable['ratio_loss_weight'] ,
    ),
    # D0=128, D1=384, 1/1 pre/post, 10 main — thin encoder/decoder, deep core
    "isotropic": HNetConfig(
        d_model=384,
        d_inner=384,
        n_head=8,
        n_pre=2,
        n_post=2,
        n_main=6,
        ratio_loss_weight=overridable['ratio_loss_weight'] ,
    ),
}

shape = overridable['shape']
assert shape in SHAPES, f"shape must be one of {list(SHAPES.keys())}, got {shape!r}"
config = SHAPES[shape]
config.vocab_size = 256
config.block_size = overridable['block_size']
config.bias = overridable['bias']
config.device = overridable['device']
config.pos_emb = overridable['pos_emb']

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
if overridable['pos_emb'] == 'rope_2d':
    if raster == 'hilbert':
        coords_list = [_hilbert_d2xy(max(H, W), d) for d in range(H * W)]
    else:
        coords_list = [(i % W, i // W) for i in range(H * W)]
    # Pad position 0 (BOS) with (0, 0)
    config.pos_coords = torch.tensor([(0, 0)] + coords_list, dtype=torch.long)

# -----------------------------------------------------------------------------#
# Init Model
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_sampled_data(overridable['dataset'], prepare_image_anime_face)

model = HNetLM(config)

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
