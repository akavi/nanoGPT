import sys
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import torch
import os

from models.model import ModuleList
from models.categorical import CategoricalConfig, Categorical
from models.mamba import Mamba2, MambaConfig
from train import train, TrainConfig
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_fixed_batch,
    get_sampled_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face

overridable = override(sys.argv, {
    "out_dir": "out-face-hilbert-raster",
    "dataset": "image_anime_face",
    "mode": "from_scratch",
    "device": "cuda",
    "seed":1337,
    "learning_rate":3e-4,
    "min_lr":3e-5,
    "n_layer": 10,
    "n_embd":384,
    "bias": False,
    "block_size": 1024,
    "max_iters": 3000,
})

# -----------------------------------------------------------------------------#
# Init Model
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_sampled_data(overridable['dataset'], prepare_image_anime_face)
backbone = ModuleList([
    Mamba2(MambaConfig(
        n_head=8,
        n_embd=overridable['n_embd'],
        n_inner=768,
        n_conv=4,
        n_state=64,

        bias=overridable['bias'],
        n_chunk=32,
        dropout=0.05,
        device=overridable['device'],
        mode="train" if overridable["mode"] in ["from_scratch", "resume"] else "sample",
    ), i)
for i in range(overridable['n_layer'])])
model = Categorical(CategoricalConfig(
    n_block=overridable['block_size'],
    n_vocab=meta['vocab_size'],
    n_embd=overridable['n_embd'],
    bias=overridable['bias'],
    dropout=0.05,
), backbone)

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

    optimizer = configure_optimizers(
        model,
        optimizer_config,
    )

elif mode == "resume" or mode == "sample":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        overridable['out_dir'], overridable['device'], model, optimizer_config,
    )
    print("Best val loss: ", best_val_loss)
else:
    raise ValueError(f"Unsupported mode={mode}")

# -----------------------------------------------------------------------------#
# Hilbert curve ordering for 32x32 grid
# -----------------------------------------------------------------------------#

BOS_ID = 0
H, W, C = 32, 32, 1
TOKENS_LINEAR = H * W * C
TOKENS_PER_ROW = TOKENS_LINEAR + 1

def _hilbert_d2xy(n, d):
    """Convert Hilbert curve index d to (x, y) in an n x n grid."""
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
    """Build permutation: hilbert_order[i] = linear index of the i-th Hilbert curve pixel."""
    n = max(h, w)
    assert n & (n - 1) == 0, "Hilbert curve requires power-of-2 grid"
    order = []
    for d in range(n * n):
        x, y = _hilbert_d2xy(n, d)
        if x < w and y < h:
            order.append(y * w + x)
    return order

# Precompute the Hilbert permutation and its inverse
HILBERT_ORDER = _build_hilbert_order(H, W)  # hilbert[i] -> linear index
INV_HILBERT_ORDER = [0] * len(HILBERT_ORDER)
for i, li in enumerate(HILBERT_ORDER):
    INV_HILBERT_ORDER[li] = i

HILBERT_ORDER_T = torch.tensor(HILBERT_ORDER, dtype=torch.long)
INV_HILBERT_ORDER_T = torch.tensor(INV_HILBERT_ORDER, dtype=torch.long)

# -----------------------------------------------------------------------------#
# Data utils
# -----------------------------------------------------------------------------#

def detokenize(tokens: torch.Tensor) -> Image.Image:
    """
    Inverse of Hilbert-curve rasterization (grayscale).
    tokens: length TOKENS_PER_ROW tensor (with BOS at front).
    Returns PIL.Image (mode 'L') of shape HxW.
    """
    t = tokens[1:].cpu()  # strip BOS
    # t is in Hilbert order — map back to linear (row-major)
    linear = torch.zeros(TOKENS_LINEAR, dtype=t.dtype)
    linear[HILBERT_ORDER_T] = t
    img = np.asarray(linear, dtype=np.uint8).reshape(H, W)
    return Image.fromarray(img, mode="L")

def get_batch(split, batch_size):
    rows = get_config_batch(
        split,
        batch_size,
        DataConfig(
            dataset=overridable['dataset'],
            device=overridable['device'],
        ),
    )
    # rows are in linear (row-major) order from the dataset — reorder to Hilbert
    idx = INV_HILBERT_ORDER_T.to(rows.device)
    rows = rows[:, idx]

    block_size = overridable['block_size']
    value = BOS_ID
    first_col = torch.full((batch_size, 1), value, dtype=rows.dtype, device=rows.device)
    x_out = torch.cat([first_col, rows[:,:-1]], dim=1).long()
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
    batch_size=128,

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
            overridable['out_dir'],
            it,
            val_loss,
            cfg,
            mdl,
            opt,
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
