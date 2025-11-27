#!/usr/bin/env python3
"""
Prebatch Anime-Face-Getchu-32x32 into [N, 1025] int16 tokens using linear (row-major) raster,
with a BOS token (0) prepended to every sample.

Outputs (next to this file):
- train.npy: np.int16, shape [N_train, 1025]
- val.npy:   np.int16, shape [N_val,   1025]
- meta.pkl:  dict describing rasterization + BOS

Notes:
- Dataset source: Kaggle -> sebastiendelprat/anime-face-getchu-32x32 (downloaded via kagglehub).
- Per-pixel tokenization order: row-major (y=0..31, x=0..31), single-channel (grayscale).
- Row format: [BOS=0, P0, P1, ..., P(1023)] where each Pi ∈ [0,255] stored as int16.
- Each row is a complete sample; no EOI; reset model state per row at training time.
"""

import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import random
import torch
from pathlib import Path

# ---- Configuration ----
BOS_ID = 0          # begin-of-sample token; intentionally collides with pixel value 0
H = 32
W = 32
C = 1               # grayscale
DATASET_SLUG = "sebastiendelprat/anime-face-getchu-32x32"  # Kaggle dataset slug
TRAIN_RATIO = 0.9

# Derived constants
TOKENS_LINEAR = H * W * C          # 1024 image tokens
TOKENS_PER_ROW = TOKENS_LINEAR + 1 # 1025 including BOS


def _list_image_files(root: Path) -> List[Path]:
    # Recursively find common image extensions
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def _tokenize_grayscale_row(im: Image) -> np.ndarray:
    """
    Load image, coerce to (H,W,1) grayscale uint8, then pack into int16 row with BOS=0.
    """
    # Force grayscale
    im = im.convert("L")
    # Ensure size HxW
    if im.size != (W, H):
        raise SystemExit(f"Unexpected size {im.size}.")
    arr = np.array(im, dtype=np.uint8)  # (H,W)
    arr = arr.reshape(H, W, 1)          # (H,W,1) for consistency
    body = arr.reshape(-1)                  # length 1024, row-major
    out = np.empty(TOKENS_PER_ROW, dtype=np.int16)
    out[0] = BOS_ID
    out[1:] = body.astype(np.int16, copy=False)
    return out


def prepare() -> (np.array, np.array, dict[str, int]): 
    # ---- Always pull from KaggleHub ----
    # Requires: pip install kagglehub ; and Kaggle credentials set up (KAGGLE_USERNAME/KAGGLE_KEY or local kaggle.json)
    print(f"Downloading dataset via kagglehub: {DATASET_SLUG} ...")
    try:
        import kagglehub
    except ImportError as e:
        raise SystemExit(
            "kagglehub is not installed. Install with `pip install kagglehub` and ensure Kaggle credentials are set."
        ) from e

    ds_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    print(f"Dataset cached at: {ds_dir}")

    # ---- Enumerate images ----
    files = _list_image_files(ds_dir)
    if not files:
        raise SystemExit(f"No images found under {ds_dir}.")
    print(f"Found {len(files)} image files.")

    # ---- Shuffle + split ----
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    train_files = files[:n_train]
    val_files   = files[n_train:]

    print(f"Total images: {n}; train: {len(train_files)} | val: {len(val_files)}")
    print(f"tokens/image (excl. BOS): {TOKENS_LINEAR} | tokens/row (incl. BOS): {TOKENS_PER_ROW}")

    # ---- Build matrices ----
    def to_matrix(paths: List[Path]) -> np.ndarray:
        m = np.empty((len(paths), TOKENS_PER_ROW), dtype=np.int16)
        for i, p in enumerate(paths):
            with Image.open(p) as im:
                m[i, :] = _tokenize_grayscale_row(im)
            if (i + 1) % 1000 == 0:
                print(f"  processed {i+1}/{len(paths)}")
        return m

    print("Building train matrix...")
    train_mat = to_matrix(train_files)
    print("train mat shape", train_mat.shape)
    print("Building val matrix...")
    val_mat = to_matrix(val_files)

    # ---- Save artifacts ----
    meta = {
        "vocab_size": 256,
    }
    return train_mat, val_mat, meta
