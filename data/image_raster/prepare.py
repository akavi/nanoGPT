#!/usr/bin/env python3
"""
Prebatch FFHQ-64x64 into [N, 12289] uint16 tokens using linear (row-major) raster,
with a BOS token (256) prepended to every sample.

Outputs (next to this file):
- train.npy: np.uint16, shape [N_train, 12289]
- val.npy:   np.uint16, shape [N_val,   12289]
- meta.pkl:  dict describing rasterization + BOS

Notes:
- Per-pixel tokenization order: row-major (y=0..63, x=0..63), channel-interleaved RGB per pixel.
- Row format: [BOS=256, R0,G0,B0, R1,G1,B1, ..., R(4095),G(4095),B(4095)].
- Each row is a complete sample; no EOI; reset model state per row at training time.
"""

import os
import pickle
from pathlib import Path
import numpy as np
from PIL import Image

BOS_ID = 256  # begin-of-sample token; requires uint16 storage
H = 64
W = 64
C = 3
dataset_name = "Dmini/FFHQ-64x64"
split_name   = "train"
seed         = 1337
train_ratio  = 0.9

def dtokenize(tokens: np.ndarray) -> np.ndarray:
    """
    Inverse of the linear row-major rasterization (RGB interleaved per pixel).
    tokens: length 12_288 array-like of ints in [0,255], NO BOS at front.
    Returns HxWx3 uint8 image.
    """
    t = np.asarray(tokens, dtype=np.uint8)
    if t.ndim != 1 or t.size != TOKENS_LINEAR:
        raise ValueError(f"expected 1D length {TOKENS_LINEAR}, got shape {t.shape}")
    img = t.reshape(H, W, C)  # row-major RGB interleaved
    return Image.fromarray(img, mode="RGB")

def print_out(img: Image):
    img.show()

def main() -> None:
    out_dir = Path(os.path.dirname(__file__))


    from datasets import load_dataset

    tokens_per_image = H * W * C          # 12_288 (image bytes only)
    tokens_per_row   = tokens_per_image+1 # 12_289 including BOS

    print(f"Loading dataset {dataset_name}:{split_name} ...")
    ds = load_dataset(dataset_name, split=split_name)

    def _to_uint16_row_linear_with_bos(ex):
        img = ex["image"]
        arr = np.array(img, copy=False)  # HWC
        arr = np.array(img, copy=False)
        if arr.ndim == 2:               # grayscale
            # keep single channel; shape HxW
            arr = arr.astype(np.uint8, copy=False)[..., None]  # HxW×1
        if arr.shape[-1] == 4:          # RGBA -> RGB
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        if arr.shape != (H, W, C):
            raise ValueError(f"Unexpected image shape {arr.shape}, expected {(H,W,C)}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # Linear row-major, channel-interleaved: arr.reshape(-1) is [R,G,B, R,G,B, ...] in raster order
        body = arr.reshape(-1)  # uint8 length 12288

        out = np.empty(tokens_per_row, dtype=np.uint16)
        out[0] = BOS_ID
        out[1:] = body.astype(np.uint16, copy=False)
        return {"row": out}

    print("Rasterizing images -> linear rows + BOS...")
    ds_rows = ds.map(
        _to_uint16_row_linear_with_bos,
        remove_columns=[c for c in ds.column_names if c != "image"],
        desc="HWC -> linear+BOS rows (uint16)",
    )

    n = len(ds_rows)
    print(f"Total images: {n}; tokens/image (excl. BOS): {tokens_per_image} | tokens/row (incl. BOS): {tokens_per_row}")

    # Shuffle + split
    ds_rows = ds_rows.shuffle(seed=seed)
    n_train = int(n * train_ratio)
    ds_train = ds_rows.select(range(n_train))
    ds_val   = ds_rows.select(range(n_train, n))

    def to_matrix(dataset):
        m = np.empty((len(dataset), tokens_per_row), dtype=np.uint16)
        for i, ex in enumerate(dataset):
            m[i, :] = ex["row"]
        return m

    print("Building train matrix...")
    train_mat = to_matrix(ds_train)
    print("Building val matrix...")
    val_mat = to_matrix(ds_val)

    # Save artifacts
    np.save(out_dir / "train.npy", train_mat, allow_pickle=False)
    np.save(out_dir / "val.npy",   val_mat,   allow_pickle=False)

    meta = {
        "dataset": dataset_name,
        "split": split_name,
        "seed": seed,
        "train_ratio": train_ratio,
        "image_height": H,
        "image_width": W,
        "channels": C,
        "raster": "row-major",
        "channel_order": "RGB",
        "channel_interleaved_per_pixel": True,
        "dtype": "uint16",
        "vocab_size": 257,              # 0..255 bytes + 256=BOS
        "bos_id": BOS_ID,
        "has_bos": True,
        "tokens_per_image": tokens_per_image,
        "tokens_per_row": tokens_per_row,  # includes BOS
        "n_train_images": int(train_mat.shape[0]),
        "n_val_images": int(val_mat.shape[0]),
        "train_file": "train.npy",
        "val_file": "val.npy",
        "note": "Linear row-major raster with leading BOS=256; each row is one full sample; no EOI.",
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved {out_dir / 'train.npy'} {train_mat.shape} {train_mat.dtype}")
    print(f"Saved {out_dir / 'val.npy'}   {val_mat.shape}   {val_mat.dtype}")
    print(f"Saved {out_dir / 'meta.pkl'}")
    print("Done.")

if __name__ == "__main__":
    main()
