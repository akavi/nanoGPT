"""
Prebatch FFHQ-64x64 into [N, 12289] uint16 tokens using a Hilbert curve order,
with a BOS token (256) prepended to every sample.

Outputs (next to this file):
- train.npy: np.uint16, shape [N_train, 12289]
- val.npy:   np.uint16, shape [N_val,   12289]
- meta.pkl:  dict describing rasterization + BOS

Notes:
- Per-visit tokenization: for each Hilbert (x,y), append [R,G,B] after a leading BOS.
- Each row is one complete sample; no EOI needed.
"""

import os
import pickle
from pathlib import Path
import numpy as np

BOS_ID = 256  # ★ begin-of-sample token; requires uint16 storage

def _hilbert_d2xy(n: int, d: int):
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - y, s - 1 - x
            else:
                x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def _hilbert_coords(n: int):
    return [_hilbert_d2xy(n, d) for d in range(n * n)]

def main() -> None:
    out_dir = Path(os.path.dirname(__file__))

    dataset_name = "Dmini/FFHQ-64x64"
    split_name   = "train"
    seed         = 1337
    train_ratio  = 0.9

    from datasets import load_dataset

    H = W = 64
    C = 3
    assert (H & (H - 1)) == 0 and H == W, "Hilbert code assumes square power-of-two grid"

    tokens_per_image = H * W * C          # 12_288 (image bytes only)
    tokens_per_row   = tokens_per_image+1 # ★ +1 for BOS -> 12_289

    print(f"Loading dataset {dataset_name}:{split_name} ...")
    ds = load_dataset(dataset_name, split=split_name)

    print("Precomputing 64x64 Hilbert order...")
    coords = _hilbert_coords(W)  # 4096 (x,y) pairs

    def _to_uint16_row_hilbert_with_bos(ex):
        img = ex["image"]
        arr = np.array(img, copy=False)  # HWC
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape != (H, W, C):
            raise ValueError(f"Unexpected image shape {arr.shape}, expected {(H,W,C)}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # ★ Allocate uint16 and prepend BOS
        out = np.empty(tokens_per_row, dtype=np.uint16)
        out[0] = BOS_ID
        j = 1
        for (x, y) in coords:
            px = arr[y, x]           # uint8[3]
            # write as uint16 to avoid implicit upcast later
            out[j:j+3] = px.astype(np.uint16, copy=False)
            j += 3
        return {"row": out}

    print("Rasterizing images -> Hilbert rows + BOS...")
    ds_rows = ds.map(
        _to_uint16_row_hilbert_with_bos,
        remove_columns=[c for c in ds.column_names if c != "image"],
        desc="HWC -> Hilbert+BOS rows (uint16)",
    )

    n = len(ds_rows)
    print(f"Total images: {n}; tokens/image (excl. BOS): {tokens_per_image} | tokens/row (incl. BOS): {tokens_per_row}")

    ds_rows = ds_rows.shuffle(seed=seed)
    n_train = int(n * train_ratio)
    ds_train = ds_rows.select(range(n_train))
    ds_val   = ds_rows.select(range(n_train, n))

    def to_matrix(dataset):
        m = np.empty((len(dataset), tokens_per_row), dtype=np.uint16)  # ★ uint16
        for i, ex in enumerate(dataset):
            m[i, :] = ex["row"]
        return m

    print("Building train matrix...")
    train_mat = to_matrix(ds_train)
    print("Building val matrix...")
    val_mat = to_matrix(ds_val)

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
        "raster": "hilbert",
        "channel_order": "RGB",
        "channel_interleaved_per_pixel": True,
        "dtype": "uint16",            # ★
        "vocab_size": 257,            # ★ 0..255 data + 256=BOS
        "bos_id": BOS_ID,             # ★
        "has_bos": True,              # ★
        "tokens_per_image": tokens_per_image,
        "tokens_per_row": tokens_per_row,  # ★ includes BOS
        "n_train_images": int(train_mat.shape[0]),
        "n_val_images": int(val_mat.shape[0]),
        "train_file": "train.npy",
        "val_file": "val.npy",
        "note": "Hilbert order with leading BOS=256; each row is one full sample; no EOI.",
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved {out_dir / 'train.npy'} {train_mat.shape} {train_mat.dtype}")
    print(f"Saved {out_dir / 'val.npy'}   {val_mat.shape}   {val_mat.dtype}")
    print(f"Saved {out_dir / 'meta.pkl'}")
    print("Done.")

if __name__ == "__main__":
    main()
