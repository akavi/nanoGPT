import sys
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
from PIL import Image
import torch
import math
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
    get_fixed_batch,
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
    "n_embd":512,
    "bias": True,
    "block_size": 1088,
    "scale_factor": 14,
    "max_iters": 3000,
    "n_mix": 10,
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
    n_mix=overridable['n_mix'],
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
W_RFFT = W // 2 + 1          # width of rfft2 output
TOKENS_LINEAR = 2 * H * W_RFFT  # *2 for (real, imag) interleaving
BOS_ID=0

# ---------- helpers: shell ordering + frequencies ----------

def _chebyshev_shell_indices(h: int, w_rfft: int) -> List[Tuple[int, int]]:
    """
    Return (i, j) indices for an h x w_rfft grid ordered by
    Chebyshev distance from the center (cy = h//2, cx = 0),
    then by squared Euclidean distance, then by (i, j).

    This gives a deterministic "spiral-ish" ordering outward
    from DC after fftshift on axis 0 only.
    """
    cy, cx = h // 2, 0  # DC index after fftshift along axis 0
    coords = [(i, j) for i in range(h) for j in range(w_rfft)]

    def key(ij: Tuple[int, int]):
        i, j = ij
        ky = i - cy      # vertical frequency index (can be negative)
        kx = j           # horizontal frequency index, rfft: kx >= 0
        r_inf = max(abs(ky), abs(kx))      # Chebyshev radius
        r2 = ky * ky + kx * kx             # Euclidean radius^2 (tie-breaker)
        return (r_inf, r2, i, j)

    coords.sort(key=key)
    return coords


def _frequency(i: int, j: int, h: int) -> float:
    """
    Euclidean frequency magnitude for index (i, j) in the
    rfft half-plane after fftshift along axis 0 only.
    """
    cy = h // 2
    ky = i - cy   # vertical frequency index

    kx = j        # horizontal frequency index (>= 0)
    return math.sqrt(ky * ky + kx * kx)


def _rasterize_shell_rfft_scaled(coeffs_shifted: np.ndarray) -> np.ndarray:
    """
    coeffs_shifted: complex array, shape (H, W_RFFT), after fftshift on axis=0.

    Returns:
        flat: float32 1D array of length TOKENS_LINEAR
              with Re/Im interleaved, AFTER scaling each
              coefficient by its frequency magnitude.
    """
    assert coeffs_shifted.shape == (H, W_RFFT), \
        f"expected {(H, W_RFFT)}, got {coeffs_shifted.shape}"

    flat = np.empty(TOKENS_LINEAR, dtype=np.float32)
    idxs = _chebyshev_shell_indices(H, W_RFFT)

    k = 0
    for i, j in idxs:
        c = coeffs_shifted[i, j]
        freq = _frequency(i, j, H)

        # multiply by frequency; leave DC (freq == 0) unscaled
        if freq != 0.0:
            c = c * freq  / (2.0**overridable['scale_factor'])
        else:
            c = c  / (2.0**overridable['scale_factor'] * 3)

        flat[k] = c.real
        flat[k + 1] = c.imag
        k += 2

    return flat


def _derasterize_shell_rfft_scaled(flat: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    flat: 1D float array of length TOKENS_LINEAR
          containing Re/Im interleaved, each having
          been multiplied by its frequency magnitude.

    h, w: original spatial image dimensions.

    Returns:
        coeffs_shifted: complex array, shape (h, w//2 + 1),
                        after *undoing* the frequency scaling,
                        still in fftshifted layout along axis 0.
    """
    w_rfft = w // 2 + 1
    expected = 2 * h * w_rfft
    assert flat.ndim == 1 and flat.size == expected, \
        f"expected size {expected}, got {flat.size}"

    coeffs_shifted = np.empty((h, w_rfft), dtype=np.complex64)
    idxs = _chebyshev_shell_indices(h, w_rfft)

    k = 0
    for i, j in idxs:
        re = flat[k]
        im = flat[k + 1]
        c = re + 1j * im
        freq = _frequency(i, j, h)

        # undo frequency scaling; leave DC unscaled
        if freq != 0.0:
            c = c  / freq * (2.0**overridable['scale_factor'])
        else:
            c = c  * (2.0**overridable['scale_factor'] * 3)

        coeffs_shifted[i, j] = c
        k += 2

    return coeffs_shifted


# ---------- main API: tokenize / detokenize ----------

def tokenize(arr: torch.Tensor) -> torch.Tensor:
    """
    Input:
        arr: torch.Tensor, either shape (H*W,) or (H, W).
             Values are interpreted as a real image.

    Output:
        1D float32 tensor of length TOKENS_LINEAR containing
        rfft2 coefficients (after fftshift on axis 0),
        Chebyshev-shell rasterized, with Re/Im interleaved,
        and each coefficient multiplied by its frequency magnitude.
    """
    # Move to numpy
    pixels = arr.detach().cpu().numpy()

    # Ensure shape (H, W)
    if pixels.ndim == 1:
        assert pixels.size == H * W, f"expected {H*W}, got {pixels.size}"
        pixels = pixels.reshape(H, W)
    else:
        assert pixels.shape == (H, W), f"expected {(H, W)}, got {pixels.shape}"

    # Use float for FFT
    img = pixels.astype(np.float32)

    # Forward real FFT: (H, W) -> (H, W_RFFT)
    coeffs = np.fft.rfft2(img)  # complex64/complex128

    # Shift only along axis 0 so DC goes to (H//2, 0)
    coeffs_shifted = np.fft.fftshift(coeffs, axes=(0,))

    # Rasterize in Chebyshev shell order and scale by frequency
    flat = np.tanh(_rasterize_shell_rfft_scaled(coeffs_shifted))

    # Back to torch
    return torch.from_numpy(flat).to(torch.float32)


def detokenize(tokens: torch.Tensor) -> torch.Tensor:
    """
    Input:
        tokens: 1D float tensor of length TOKENS_LINEAR.

        Tokens are assumed to come from `tokenize`, i.e.,
        Chebyshev-shell rasterized rfft2 coefficients with
        Re/Im interleaved and multiplied by frequency magnitude.

    Output:
        1D float32 tensor of length H*W with reconstructed image.
    """
    coeffs_flat = np.arctanh(tokens.detach().cpu().numpy().astype(np.float32))
    expected = TOKENS_LINEAR
    assert coeffs_flat.ndim == 1 and coeffs_flat.size == expected, \
        f"actual dim={coeffs_flat.ndim}, actual size={coeffs_flat.size}, expected={expected}"

    # Recover shifted complex spectrum and undo frequency scaling
    coeffs_shifted = _derasterize_shell_rfft_scaled(coeffs_flat, H, W)

    # Undo fftshift along axis 0
    coeffs = np.fft.ifftshift(coeffs_shifted, axes=(0,))

    # Inverse real FFT back to spatial domain
    img = np.fft.irfft2(coeffs, s=(H, W))  # shape (H, W), real

    # Keep as float32 to avoid extra quantization
    img = img.astype(np.float32)

    # Flatten back to 1D
    return torch.from_numpy(img.reshape(H * W))


# Length TOKENS_LINEAR; freq duplicated for (Re, Im) for each (i, j)
_FREQS_LINEAR_CPU = torch.tensor(
    [
        f
        for (i, j) in _chebyshev_shell_indices(H, W_RFFT)
        for f in (_frequency(i, j, H), _frequency(i, j, H))
    ],
    dtype=torch.float32,
)

def get_batch(split, batch_size):
    rows = get_fixed_batch(
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
    x_out = torch.cat([first_col, tokens[:, :-1]], dim=1)  # unchanged: (B, T)

    # Build y_out: (B, T, 2) = (value, freq)
    freqs = _FREQS_LINEAR_CPU.to(device=tokens.device, dtype=tokens.dtype)  # (T,)
    freqs_bt = freqs.unsqueeze(0).expand(batch_size, -1)                    # (B, T)
    y_out = torch.stack([tokens, freqs_bt], dim=-1)                         # (B, T, 2)

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
        return torch.zeros((1, 1), dtype=torch.bfloat16, device=device)

    def detokenize_and_save(tokens: np.ndarray, path: str):
        img = detokenize(tokens[1:]).reshape(H, W).cpu().numpy().astype(np.uint8)
        img = Image.fromarray(img, mode="L")
        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=1088,
        temperature=1,
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
