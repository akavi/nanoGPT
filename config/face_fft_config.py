import sys
from pathlib import Path
from typing import List, Tuple, Any
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
    debug_one_image,
    get_fixed_batch as get_config_batch,
    get_raw_rows,
    get_sampled_batch,
    plot_log_token_position_means,
    save_checkpoint as save_config_checkpoint,
    check_roundtrip,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
    view_roundtrip,
    view_roundtrip_once,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face
from data_utils.mdct import mdct_forward, mdct_backward

overridable = override(sys.argv, {
    "out_dir": "out-face-mdct-zigzag",
    "dataset": "image_anime_face",
    "mode": "from_scratch",  
    "device": "cuda",
    "seed":1337,
    "learning_rate":3e-5,
    "min_lr":3e-6,
    "n_layer": 10,
    "n_embd":384,
    "bias": True,
    "block_size": 1088,
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
    n_mix=10,
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

H, W = 32, 32
W_RFFT = W // 2 + 1                 # width of rfft2 output
TOKENS_LINEAR = 2 * H * W_RFFT      # *2 for (real, imag) interleaving
BOS_ID = 0

def _chebyshev_shell_indices(h: int, w_rfft: int) -> List[Tuple[int, int]]:
    cy, cx = h // 2, 0

    coords = [(i, j) for i in range(h) for j in range(w_rfft)]

    def key(ij: Tuple[int, int]):
        i, j = ij
        dy = i - cy
        dx = j - cx
        r_inf = max(abs(dy), abs(dx))   # Chebyshev "radius"
        r2 = dy * dy + dx * dx         # tie-breaker using Euclidean radius
        return (r_inf, r2, i, j)

    coords.sort(key=key)
    return coords


def _rasterize_shell_rfft(coeffs: np.ndarray) -> np.ndarray:
    assert coeffs.shape == (H, W_RFFT), f"expected {(H, W_RFFT)}, got {coeffs.shape}"

    flat = np.empty(TOKENS_LINEAR, dtype=np.float32)
    idxs = _chebyshev_shell_indices(H, W_RFFT)

    k = 0
    for i, j in idxs:
        c = coeffs[i, j]
        flat[k] = c.real
        flat[k + 1] = c.imag
        k += 2

    return flat


def _derasterize_shell_rfft(flat: np.ndarray, h: int, w: int) -> np.ndarray:
    w_rfft = w // 2 + 1
    expected = 2 * h * w_rfft

    assert flat.ndim == 1 and flat.size == expected, \
        f"expected size {expected}, got {flat.size}"

    coeffs = np.empty((h, w_rfft), dtype=np.complex64)
    idxs = _chebyshev_shell_indices(h, w_rfft)

    k = 0
    for i, j in idxs:
        re = flat[k]
        im = flat[k + 1]
        coeffs[i, j] = re + 1j * im
        k += 2

    return coeffs


def tokenize(arr: torch.Tensor) -> torch.Tensor:
    pixels = arr.detach().cpu().numpy()

    if pixels.ndim == 1:
        assert pixels.size == H * W, f"expected {H*W}, got {pixels.size}"
        pixels = pixels.reshape(H, W)
    else:
        assert pixels.shape == (H, W), f"expected {(H, W)}, got {pixels.shape}"

    img = pixels.astype(np.float32)
    coeffs = np.fft.rfft2(img)
    coeffs_shifted = np.fft.fftshift(coeffs, axes=(0,))
    flat = _rasterize_shell_rfft(coeffs_shifted)
    N, = flat.shape
    idx = torch.arange(1, N + 1, device=flat.device).cpu().numpy()   # [1, 2, ..., N]
    flat = flat * idx / 2**18
    return torch.from_numpy(flat).to(torch.float32)


def detokenize(tokens: torch.Tensor) -> torch.Tensor:
    coeffs_flat = tokens.detach().cpu().numpy().astype(np.float32)

    expected = TOKENS_LINEAR
    assert coeffs_flat.ndim == 1 and coeffs_flat.size == expected, \
        f"actual dim={coeffs_flat.ndim}, actual size={coeffs_flat.size}, expected={expected}"
    N, = coeffs_flat.shape
    idx = torch.arange(1, N + 1, device="cpu").numpy()   # [1, 2, ..., N]
    coeffs_flat = coeffs_flat / idx * 2**18
    coeffs_shifted = _derasterize_shell_rfft(coeffs_flat, H, W)
    coeffs = np.fft.ifftshift(coeffs_shifted, axes=(0,))
    img = np.fft.irfft2(coeffs, s=(H, W))   # real-valued float64
    pixels = img.astype(np.float32)
    print("DETOK SHAPE", pixels.shape)
    return torch.from_numpy(pixels.reshape(H * W))

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
# -----------------------------------------------------------------------------#"
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
        img = detokenize(tokens[1:]).reshape(H, W).cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img, mode="L")
        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=1088,
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
