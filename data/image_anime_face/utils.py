"""Shared utilities for linear-raster anime face image configs."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils import DataConfig, get_sampled_batch

BOS_ID = 0
H, W, C = 32, 32, 1
TOKENS_LINEAR = H * W * C


def detokenize(tokens: torch.Tensor) -> Image.Image:
    """Inverse of linear row-major rasterization (grayscale).
    tokens: 1D with BOS at front, ints in [0,255]. Returns PIL Image (mode 'L').
    """
    t = np.asarray(tokens[1:].cpu(), dtype=np.uint8)
    if t.ndim != 1 or t.size != TOKENS_LINEAR:
        raise ValueError(f"expected 1D length {TOKENS_LINEAR}, got shape {t.shape}")
    return Image.fromarray(t.reshape(H, W), mode="L")


def make_get_batch(dataset: str, device: str, block_size: int):
    """Return a get_batch(split, batch_size) closure for linear-raster face data."""
    config = DataConfig(dataset=dataset, device=device)

    def get_batch(split, batch_size):
        rows = get_sampled_batch(split, batch_size, config)
        first_col = torch.full((batch_size, 1), BOS_ID, dtype=rows.dtype, device=rows.device)
        x = torch.cat([first_col, rows[:, :-1]], dim=1).long()
        y = rows[:, 0:block_size + 1].contiguous().long()
        return x, y

    return get_batch


def init_gen(device):
    """Initial generation tensor (single BOS token)."""
    return torch.zeros((1, 1), dtype=int, device=device)


def save_image(tokens: np.ndarray, path: str, out_dir: str):
    """Detokenize and save a generated sample as PNG."""
    img = detokenize(tokens)
    path = os.path.join(out_dir, str(Path(path).with_suffix(".png")))
    img.save(path, format="PNG")
