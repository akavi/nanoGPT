import sys
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt  # add this near the top of your file
import numpy as np
from PIL import Image
import torch
import os

from models.model import ModuleList
from models.mol import MoLConfig, MoL
from models.mamba import Mamba2, MambaConfig
from train import train, TrainConfig
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_fixed_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face
from data_utils.mdct import mdct_forward, mdct_backward

overridable = override(sys.argv, {
    "out_dir": "out-face-mdct-zigzag",
    "dataset": "image_anime_face",
    "mode": "from_scratch",  
    "device": "cuda",
    "seed":1337,
    "learning_rate":3e-6,
    "min_lr":3e-7,
    "n_layer": 10,
    "n_embd":384,
    "bias": True,
    "block_size": 1024,
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

        bias=False,
        n_chunk=32,
        dropout=0.05,
        device=overridable['device'],
        mode="train" if overridable["mode"] in ["from_scratch", "resume"] else "sample",
    ), i)
for i in range(overridable['n_layer'])])
model = MoL(MoLConfig(
    n_block=overridable['block_size'],
    n_embd=overridable['n_embd'],
    n_mix=5,
    bias=overridable['bias'],
    dropout=0.05,
), backbone)

optimizer_config = OptimizerConfig(
    weight_decay=1e-1,
    learning_rate=overridable['learning_rate'],
    betas=[0.9, 0.95],
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
# Data utils
# -----------------------------------------------------------------------------#

# TODO: We should store raw images
# and perform conversions on the fly
BOS_ID = 0
H, W, C = 32, 32, 1
TOKENS_LINEAR = H * W * C
TOKENS_PER_ROW = TOKENS_LINEAR + 1

def _taxicab_shell_indices(h: int, w: int, boustrophedon: bool = True):
    """
    Generate (row, col) indices covering an h×w grid in 'shell' order.

    Shell r consists of all (row, col) with max(row, col) == r, row,col >= 0.

    For boustrophedon=True, the order matches your example:

      r = 0:
        (0, 0)

      r = 1 (odd):
        (1, 0), (1, 1), (0, 1)

      r = 2 (even):
        (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)

      and so on.

    For boustrophedon=False, every shell is oriented like r=1:
      start at (r, 0), along row r left→right, then up along col r.
    """

    max_r = max(h, w) - 1
    indices: list[tuple[int, int]] = []

    for r in range(max_r + 1):
        if r == 0:
            if h > 0 and w > 0:
                indices.append((0, 0))
            continue

        if boustrophedon:
            if r % 2 == 1:
                # odd r: like your r=1 example
                pts = []
                # bottom edge: (r, 0..r)
                for c in range(0, r + 1):
                    pts.append((r, c))
                # right edge: (r-1..0, r)
                for i in range(r - 1, -1, -1):
                    pts.append((i, r))
            else:
                # even r: like your r=2 example
                pts = []
                # right edge: (0..r, r)
                for i in range(0, r + 1):
                    pts.append((i, r))
                # bottom edge: (r, r-1..0)
                for c in range(r - 1, -1, -1):
                    pts.append((r, c))
        else:
            # non-boustrophedon: always like r=1 orientation
            pts = []
            # bottom edge: (r, 0..r)
            for c in range(0, r + 1):
                pts.append((r, c))
            # right edge: (r-1..0, r)
            for i in range(r - 1, -1, -1):
                pts.append((i, r))

        # clip to image bounds
        for i, j in pts:
            if 0 <= i < h and 0 <= j < w:
                indices.append((i, j))

    # Optional sanity check:
    # assert len(indices) == h * w and len(set(indices)) == h * w
    return indices


def _rasterize_shell(img: np.ndarray, boustrophedon: bool = True) -> np.ndarray:
    """
    Rasterize a 2-D array into 1-D using taxicab shells.
    """
    if img.ndim != 2:
        raise ValueError("_rasterize_shell expects a 2-D array")
    h, w = img.shape
    order = _taxicab_shell_indices(h, w, boustrophedon=boustrophedon)
    out = np.empty(h * w, dtype=img.dtype)
    for k, (i, j) in enumerate(order):
        out[k] = img[i, j]
    return out


def _derasterize_shell(vec: np.ndarray, h: int, w: int, boustrophedon: bool = True) -> np.ndarray:
    """
    Inverse of _rasterize_shell with the same boustrophedon flag.
    """
    if vec.ndim != 1:
        raise ValueError("_derasterize_shell expects a 1-D array")
    if vec.size != h * w:
        raise ValueError(f"vector length {vec.size} does not match h*w={h*w}")

    order = np.array(_taxicab_shell_indices(h, w, boustrophedon=boustrophedon))  # (h*w, 2)
    out = np.empty((h, w), dtype=vec.dtype)
    out[order[:, 0], order[:, 1]] = vec
    return out

def _zigzag_indices(h, w):
    pairs = [(u,v) for u in range(h) for v in range(w)]
    return sorted(pairs, key=lambda t: (t[0]+t[1], t[0]))

def _rasterize(img: np.ndarray) -> np.ndarray:
    """
    Return a 1-D vector of img's elements in zigzag (JPEG) order.
    img must be 2-D (grayscale or single-channel coefficient plane).
    """
    if img.ndim != 2:
        raise ValueError("rasterize expects a 2-D array")
    h, w = img.shape
    out = np.empty(h * w, dtype=img.dtype)
    for index, (i, j) in enumerate(_zigzag_indices(h, w)):
        out[index] = img[i, j]
    return out

def _derasterize(vec: np.ndarray, h: int, w: int) -> np.ndarray:
    if vec.ndim != 1:
        raise ValueError("_derasterize expects a 1-D array")
    if vec.size != h * w:
        raise ValueError(f"vector length {vec.size} does not match h*w={h*w}")

    out = np.empty((h, w), dtype=vec.dtype)
    ij = np.array(_zigzag_indices(h, w))  # shape: (h*w, 2)
    out[ij[:, 0], ij[:, 1]] = vec
    return out

def tokenize(arr: torch.Tensor) -> torch.Tensor:
    pixels = arr.detach().cpu().numpy().astype(np.uint8)
    img = pixels.reshape(H, W)              # (32, 32)
    coeffs = mdct_forward(img) 
    flat = _rasterize_shell(coeffs) 
    N, = flat.shape
    idx = torch.arange(1, N + 1, device=flat.device).cpu().numpy()   # [1, 2, ..., N]
    flat = flat * idx / 2**18
    return torch.from_numpy(flat).to(torch.bfloat16) 

def detokenize(tokens: torch.Tensor) -> torch.Tensor:
    coeffs_flat = tokens.detach().cpu().numpy()
    assert coeffs_flat.ndim == 1 and coeffs_flat.size == TOKENS_LINEAR, f"actual dim={coeffs_flat.ndim}, actual size={coeffs_flat.size}"
    N, = coeffs_flat.shape
    idx = torch.arange(1, N + 1, device="cpu").numpy()   # [1, 2, ..., N]
    coeffs_flat = coeffs_flat / idx * 2**18
    coeffs = _derasterize_shell(coeffs_flat, H, W).astype(np.int32)                        # (H, W)
    pixels = mdct_backward(coeffs) 
    return torch.from_numpy(pixels.reshape(H*W))

def get_batch(split, batch_size):
    rows = get_config_batch(
        split,
        batch_size,
        DataConfig(
            dataset=overridable['dataset'],
            device=overridable['device'],
        ),
    )
    tokens = torch.stack(
        [tokenize(row) for row in rows],
        dim=0,
    ).to(overridable['device'])              # (B, TOKENS_PER_ROW), float32

    value = BOS_ID
    first_col = torch.full((batch_size, 1), value, dtype=tokens.dtype, device=tokens.device)
    x_out = torch.cat([first_col, tokens[:,:-1]], dim=1)
    y_out = tokens
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
    batch_size=1,                

    eval_only=False,
    eval_interval=250,
    eval_iters=20,
    warmup_iters=300,
    max_iters=3000,

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
        return torch.zeros((1, 1), dtype=torch.bfloat16, device=device)

    def detokenize_and_save(tokens: np.ndarray, path: str):
        img = detokenize(tokens[1:]).reshape(H, W).cpu().numpy()
        img = Image.fromarray(img, mode="L")
        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=1024,
        temperature=0.8,
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
