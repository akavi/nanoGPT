#!/usr/bin/env python3
"""
Prebatch Anime-Face-Getchu-32x32 into [N, 1025] uint8 tokens using linear (row-major) raster,
with a BOS token (0) prepended to every sample.

Outputs (next to this file):
- train.npy: np.uint8, shape [N_train, 1025]
- val.npy:   np.uint8, shape [N_val,   1025]
- meta.pkl:  dict describing rasterization + BOS

Notes:
- Dataset source: Kaggle -> sebastiendelprat/anime-face-getchu-32x32 (downloaded via kagglehub).
- Per-pixel tokenization order: row-major (y=0..31, x=0..31), single-channel (grayscale).
- Row format: [BOS=0, P0, P1, ..., P(1023)] where each Pi ∈ [0,255] stored as uint8.
- Each row is a complete sample; no EOI; reset model state per row at training time.
"""

import os
import pickle
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import random
from data_utils.mdct import mdct_forward, mdct_backward
import sys

# ---- Configuration ----
BOS_ID = 0          # begin-of-sample token; intentionally collides with pixel value 0
H = 32
W = 32
C = 1               # grayscale
DATASET_SLUG = "sebastiendelprat/anime-face-getchu-32x32"  # Kaggle dataset slug
SEED = 1337
TRAIN_RATIO = 0.9

# Derived constants
WIDTH = 4
TOKENS_LINEAR = WIDTH*H * W * C          # 1024 image tokens
TOKENS_PER_ROW = TOKENS_LINEAR + 1 # 1025 including BOS

def detokenize(tokens: np.ndarray, path) -> Image.Image:
    """
    Inverse of the linear row-major rasterization (grayscale).
    tokens: length TOKENS_LINEAR array-like of ints in [0,255], NO BOS at front.
    Returns PIL.Image (mode 'L') of shape HxW.
    """
    t = np.asarray(tokens, dtype=np.uint8)
    if t.ndim != 1 or t.size != TOKENS_LINEAR:
        raise ValueError(f"expected 1D length {TOKENS_LINEAR}, got shape {t.shape}")

    img = mdct_backward(_derasterize(i8_to_i32(tokens), H, W))
    img = Image.fromarray(img, mode="L")
    img.save(str(Path(path).with_suffix(".png")), format="PNG")

def _derasterize(vec: np.ndarray, h: int, w: int) -> np.ndarray:
    if vec.ndim != 1:
        raise ValueError("_derasterize expects a 1-D array")
    if vec.size != h * w:
        raise ValueError(f"vector length {vec.size} does not match h*w={h*w}")

    out = np.empty((h, w), dtype=vec.dtype)
    ij = np.array(_zigzag_indices(h, w))  # shape: (h*w, 2)
    out[ij[:, 0], ij[:, 1]] = vec
    return out

def print_out(img: Image.Image):
    img.show()

def _list_image_files(root: Path) -> List[Path]:
    # Recursively find common image extensions
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

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

def i32_to_u8(x: np.ndarray, byteorder: str = "little", copy: bool = True) -> np.ndarray:
    a = np.ascontiguousarray(x, dtype=np.int32)  # ensure C-contiguous int32
    if byteorder != "native":
        want_le = (byteorder == "little")
        native_le = (sys.byteorder == "little")
        if want_le != native_le:
            # swap per-32-bit word to produce the requested ordering
            a = a.byteswap().newbyteorder()

    out = a.view(np.uint8)  # reinterpret bytes
    return out.copy() if copy else out  # optional copy for stability

def u8_to_i32(b: np.ndarray, byteorder: str = "little", copy: bool = True) -> np.ndarray:
    bb = np.ascontiguousarray(b, dtype=np.uint8)
    if bb.size % 4 != 0:
        raise ValueError("uint8 length must be a multiple of 4")

    out = bb.view(np.int32)  # reinterpret bytes into 32-bit words

    if byteorder != "native":
        want_le = (byteorder == "little")
        native_le = (sys.byteorder == "little")
        if want_le != native_le:
            out = out.byteswap().newbyteorder()

    return out.copy() if copy else out

def _load_and_tokenize_grayscale_row(path: Path) -> np.ndarray:
    """
    Load image, coerce to (H,W,1) grayscale uint8, then pack into uint8 row with BOS=0.
    """
    with Image.open(path) as im:
        # Force grayscale
        im = im.convert("L")
        # Ensure size HxW
        if im.size != (W, H):
            raise SystemExit(f"Unexpected size {im.size}.")
        arr = np.array(im, dtype=np.uint8)  # (H,W)
        arr = arr.reshape(H, W)          # (H,W,1) for consistency
    body = i32_to_u8(_rasterize(mdct_forward(arr)))
    out = np.empty(TOKENS_PER_ROW, dtype=np.uint8)
    out[0] = BOS_ID
    out[1:] = body
    return out


def main() -> None:
    out_dir = Path(os.path.dirname(__file__))

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
    rng = random.Random(SEED)
    files_sorted = sorted(files)  # deterministic base order before shuffle
    rng.shuffle(files_sorted)
    n = len(files_sorted)
    n_train = int(n * TRAIN_RATIO)
    train_files = files_sorted[:n_train]
    val_files   = files_sorted[n_train:]

    print(f"Total images: {n}; train: {len(train_files)} | val: {len(val_files)}")
    print(f"tokens/image (excl. BOS): {TOKENS_LINEAR} | tokens/row (incl. BOS): {TOKENS_PER_ROW}")

    # ---- Build matrices ----
    def to_matrix(paths: List[Path]) -> np.ndarray:
        m = np.empty((len(paths), TOKENS_PER_ROW), dtype=np.uint8)
        for i, p in enumerate(paths):
            m[i, :] = _load_and_tokenize_grayscale_row(p)
            if (i + 1) % 1000 == 0:
                print(f"  processed {i+1}/{len(paths)}")
        return m

    print("Building train matrix...")
    train_mat = to_matrix(train_files)
    print("Building val matrix...")
    val_mat = to_matrix(val_files)

    # ---- Save artifacts ----
    np.save(out_dir / "train.npy", train_mat, allow_pickle=False)
    np.save(out_dir / "val.npy",   val_mat,   allow_pickle=False)

    meta = {
        "dataset": DATASET_SLUG,
        "source": "kagglehub",
        "split": {"train_ratio": TRAIN_RATIO, "seed": SEED},
        "vocab_size": 256,
        "dtype": "uint8",
        "note": "BOS=0 collides with pixel value 0 by design; effective pixel alphabet 0..255. Each row is one full sample; no EOI; reset model state per row.",
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved {out_dir / 'train.npy'} {train_mat.shape} {train_mat.dtype}")
    print(f"Saved {out_dir / 'val.npy'}   {val_mat.shape}   {val_mat.dtype}")
    print(f"Saved {out_dir / 'meta.pkl'}")
    print("Done.")


if __name__ == "__main__":
    main()
