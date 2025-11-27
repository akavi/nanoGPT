import sys
from pathlib import Path
from typing import Any
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
    get_sampled_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face
from data_utils.mdct import mdct_forward, mdct_backward

overridable = override(sys.argv, {
    "out_dir": "out-face-linear-raster",
    "dataset": "image_anime_face",
    "mode": "from_scratch",  
    "device": "cuda",
    "seed":1337,
    "learning_rate":3e-4,
    "min_lr":3e-5,
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
# Data utils
# -----------------------------------------------------------------------------#

# TODO: We should store raw images
# and perform conversions on the fly
BOS_ID = 0
H, W, C = 32, 32, 1
TOKENS_LINEAR = H * W * C
TOKENS_PER_ROW = TOKENS_LINEAR + 1

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
    arr = arr.detach().cpu().numpy()
    pixels = arr[1:].astype(np.uint8)  # (1024,)
    img = pixels.reshape(H, W)              # (32, 32)
    coeffs = mdct_forward(img)              # (H, W) float32
    flat = _rasterize(coeffs)               # (1024,)
    out = np.empty(TOKENS_PER_ROW, dtype=np.float32)
    out[0] = float(BOS_ID)
    out[1:] = flat
    return torch.from_numpy(out)

def detokenize(tokens: torch.Tensor) -> Image.Image:
    # tokens: (..., TOKENS_PER_ROW) torch.float*
    t = tokens.detach().cpu().numpy()
    assert t.ndim == 1 and t.size == TOKENS_PER_ROW
    coeffs_flat = t[1:].astype(np.int32)                         
    coeffs = _derasterize(coeffs_flat, H, W)    # (H, W)
    img = mdct_backward(coeffs)
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
    tokens = torch.stack(
        [tokenize(row) for row in rows],
        dim=0,
    ).to(overridable['device'])              # (B, TOKENS_PER_ROW), float32

    block_size = overridable['block_size']
    x_out = tokens[:, :block_size]           # (B, 1024)
    y_out = tokens[:, 1:block_size + 1].contiguous()      # (B, 1024)

    return x_out, y_out

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

train_config = TrainConfig(
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    device=overridable['device'],

    decay_lr=True,
    lr_decay_iters=300,
    learning_rate=overridable['learning_rate'],
    min_lr=overridable['min_lr'],

    grad_clip=1.0,
    gradient_accumulation_steps=1,
    batch_size=128,                # also used in get_batch

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
        return torch.zeros((1, 1), dtype=int, device=device)

    def detokenize_and_save(tokens: np.ndarray, path: str):
        img = detokenize(tokens)
        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=500,
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
