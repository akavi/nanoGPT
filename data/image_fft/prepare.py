"""
Prebatch FFHQ-64x64 into [N, T] uint16 tokens representing byte-split
Fourier coefficients in a spiral order, with BOS=256.

Ordering per frequency bin:
  Real(R), Real(G), Real(B), Imag(R), Imag(G), Imag(B)
Each Real/Imag is stored as little-endian float32 -> 4 bytes each.

Spatial ordering:
  fftshift-centered, unique half-plane (drop Hermitian duplicates):
    keep iff (ky > 0) or (ky == 0 and kx >= 0)
  Within kept bins: sort by radius^2 ascending, then atan2(ky, kx) ascending.
  (This yields a deterministic "spiral out from origin" by shells.)

Row format (uint16):
  [256 (BOS), byte0, byte1, ..., byteM]   where bytes are 0..255

Outputs (next to this file):
  - train.npy (uint16, shape [N_train, T])
  - val.npy   (uint16, shape [N_val,   T])
  - meta.pkl  (settings + T)
"""

import os
import pickle
from pathlib import Path
import numpy as np
from PIL import Image

BOS_ID = 256
H = 28
W = 28
C = 1
dataset_name = "ylecun/mnist"
split_name   = "train"
seed         = 1337
train_ratio  = 0.9

def _unique_halfplane_coords(n: int):
    """
    Recreate the fftshift-based unique coordinates we used when encoding:
      keep iff (ky > 0) or (ky == 0 and kx >= 0)
    Returned as (y_idx, x_idx) in *fftshifted index space* (0..n-1).
    """
    # Construct ky,kx values in natural math coords, then sort by (r^2, angle),
    # then map to fftshifted indices.
    ky_vals = np.arange(-n//2, n//2, dtype=np.int32)  # -32..31
    kx_vals = ky_vals.copy()
    ky, kx = np.meshgrid(ky_vals, kx_vals, indexing='ij')
    keep = (ky > 0) | ((ky == 0) & (kx >= 0))
    ky_kept = ky[keep]
    kx_kept = kx[keep]
    r2 = ky_kept.astype(np.int64)**2 + kx_kept.astype(np.int64)**2
    ang = np.arctan2(ky_kept, kx_kept)
    order = np.lexsort((ang, r2))  # primary r2, secondary angle
    ky_sorted = ky_kept[order]
    kx_sorted = kx_kept[order]
    # map to fftshifted indices: index = coord + n//2
    y_idx = (ky_sorted + n//2).astype(np.int64)
    x_idx = (kx_sorted + n//2).astype(np.int64)
    return y_idx, x_idx  # length K

# cached
_YI, _XI = _unique_halfplane_coords(W)
_K = _YI.shape[0]

_BYTES_PER_FREQ = C * 2 * 4  #  numChannels * [Re/Im] * float32

def detokenize(tokens: np.ndarray, norm: str | None = 'ortho') -> Image:
    """
    Inverse of the FFT-byte serialization (drop Hermitian duplicates).
    tokens: 1D uint8/uint16 array of length K*24 (NO BOS) where each kept bin stores:
            Re(R),Re(G),Re(B), Im(R),Im(G),Im(B), each as little-endian float32.
    norm: 'ortho' or None -- MUST match what you used when encoding.

    Returns HxWx3 uint8 image.
    """
    t = np.asarray(tokens)
    if t.ndim != 1:
        raise ValueError(f"expected 1D token array, got shape {t.shape}")
    # Accept uint8 or uint16 holding 0..255
    if t.dtype == np.uint16:
        if np.any(t > 255):
            raise ValueError("fft-byte stream should be pure 0..255 bytes (no BOS)")
        bytes_arr = t.astype(np.uint8, copy=False)
    elif t.dtype == np.uint8:
        bytes_arr = t
    else:
        bytes_arr = t.astype(np.uint8, copy=False)

    expected_len = _K * _BYTES_PER_FREQ
    if bytes_arr.size != expected_len:
        raise ValueError(f"length {bytes_arr.size} != expected {expected_len} (K={_K}, bytes/freq={_BYTES_PER_FREQ})")

    # View as little-endian float32 blocks of 6 per frequency
    # Reshape [K, 24] -> view as [K, 6] float32
    blocks = bytes_arr.reshape(_K, _BYTES_PER_FREQ).view('<f4')  # shape [K, 6]
    # Split into Re/Im per channel
    ReR, ReG, ReB, ImR, ImG, ImB = (blocks[:, i] for i in range(6))

    # Allocate fftshifted spectra (complex64) per channel
    F = [np.zeros((H, W), dtype=np.complex64) for _ in range(C)]

    # Fill kept bins
    for k, (yi, xi) in enumerate(zip(_YI.tolist(), _XI.tolist())):
        F[0][yi, xi] = np.complex64(ReR[k] + 1j * ImR[k])
        F[1][yi, xi] = np.complex64(ReG[k] + 1j * ImG[k])
        F[2][yi, xi] = np.complex64(ReB[k] + 1j * ImB[k])

    # Fill the dropped half by conjugate symmetry in fftshifted index space.
    # For a kept index (yi,xi), the symmetric is (n-yi)%n, (n-xi)%n.
    n = H
    for c in range(C):
        Fc = F[c]
        # Mirror fill
        for yi, xi in zip(_YI.tolist(), _XI.tolist()):
            yj = (n - yi) % n
            xj = (n - xi) % n
            # If it's not the same bin (DC or certain boundaries), write the conj.
            if (yj != yi) or (xj != xi):
                Fc[yj, xj] = np.conj(Fc[yi, xi])
        F[c] = Fc

    # Inverse shift + iFFT to spatial
    img = np.empty((H, W, C), dtype=np.float32)
    for c in range(C):
        Fc_unshift = np.fft.ifftshift(F[c])
        x_space = np.fft.ifft2(Fc_unshift, norm=norm)
        img[..., c] = np.real(x_space).astype(np.float32)

    # Clip to [0,255] and convert to uint8
    img_u8 = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")

def print_out(img: Image):
    img.show()

def _unique_halfplane_coords(n):
    ky_vals = np.arange(-n//2, n//2, dtype=np.int32)
    kx_vals = ky_vals.copy()
    ky, kx = np.meshgrid(ky_vals, kx_vals, indexing='ij')
    keep = (ky > 0) | ((ky == 0) & (kx >= 0))
    ky_kept = ky[keep]; kx_kept = kx[keep]

    # spiral-ish: sort by radius^2 then angle
    r2 = ky_kept.astype(np.int64)**2 + kx_kept.astype(np.int64)**2
    ang = np.arctan2(ky_kept, kx_kept)
    order = np.lexsort((ang, r2))
    ky_sorted = ky_kept[order]
    kx_sorted = kx_kept[order]
    y_idx = (ky_sorted + n//2).astype(np.int64)
    x_idx = (kx_sorted + n//2).astype(np.int64)
    return y_idx, x_idx  # length K

def _image_to_fft_bytes_row(arr_hwc_u8, y_idx, x_idx, norm='ortho'):
    H, W, C = arr_hwc_u8.shape
    f = arr_hwc_u8.astype(np.float32, copy=False)
    # fftshifted complex64 per channel
    F = [np.fft.fftshift(np.fft.fft2(f[..., c], norm=norm)).astype(np.complex64, copy=False)
         for c in range(C)]
    K = y_idx.shape[0]
    bytes_per_freq = (C + C) * 4  # Re for all C, then Im for all C, float32
    total_bytes = K * bytes_per_freq
    out = np.empty(1 + total_bytes, dtype=np.uint16); out[0] = 256
    byte_buf = np.empty(total_bytes, dtype=np.uint8)
    p = 0
    for yi, xi in zip(y_idx.tolist(), x_idx.tolist()):
        re = [np.float32(np.real(F[c][yi, xi])) for c in range(C)]
        im = [np.float32(np.imag(F[c][yi, xi])) for c in range(C)]
        pack = np.array(re + im, dtype='<f4')  # little-endian
        b = pack.view(np.uint8)  # 8*C bytes
        byte_buf[p:p+bytes_per_freq] = b; p += bytes_per_freq
    out[1:] = byte_buf.astype(np.uint16)
    return out

def main() -> None:
    out_dir = Path(os.path.dirname(__file__))


    from datasets import load_dataset, Features, Sequence, Value 

    print(f"Loading dataset {dataset_name}:{split_name} ...")
    ds = load_dataset(dataset_name, split=split_name)

    # Precompute unique kept coords in spiral order (fftshifted indices)
    print("Preparing unique half-plane spiral coordinates...")
    y_idx, x_idx = _unique_halfplane_coords(W)  # length K
    K = int(len(y_idx))
    bytes_per_freq = 2 * 4 * C # [Re/Im] x 3 channels x float32
    tokens_per_row = 1 + K * bytes_per_freq  # 1 for BOS; rest are bytes 0..255
    print(f"Kept frequencies: {K} (of {W*W}); tokens_per_row (incl. BOS): {tokens_per_row}")


    def _to_uint16_row_fft_bytes(ex):
        img = ex["image"]
        arr = np.array(img, copy=False)
        if arr.ndim == 2:               # grayscale
            # keep single channel; shape HxW
            arr = arr.astype(np.uint8, copy=False)[..., None]  # HxW×1
        if arr.shape[-1] == 4:          # RGBA -> RGB
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        row = _image_to_fft_bytes_row(arr, y_idx, x_idx)  # uint16 length tokens_per_row
        return {"row": row}

    print("Transforming images -> FFT-byte rows with BOS...")
    ds_rows = ds.map(
        _to_uint16_row_fft_bytes,
        remove_columns=[c for c in ds.column_names if c != "image"],
        desc="HWC -> FFT bytes (uint16)",
    )

    n = len(ds_rows)
    print(f"Total images: {n}; tokens/row: {tokens_per_row}")

    # Shuffle + split
    ds_rows = ds_rows.shuffle(seed=seed)
    n_train = int(n * train_ratio)
    ds_train = ds_rows.select(range(n_train))
    ds_val   = ds_rows.select(range(n_train, n))

    def to_matrix(dataset):
        m = np.empty((len(dataset), tokens_per_row), dtype=np.uint16)
        for i, ex in enumerate(dataset):
            r = np.asarray(ex["row"], dtype=np.uint16)  # ★ list → ndarray
            if r.size != tokens_per_row:
                raise RuntimeError(f"Row length mismatch: {r.size} vs {tokens_per_row}")
            m[i, :] = r
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
        "transform": "fft2+fftshift",
        "unique_halfplane_rule": "ky>0 or (ky==0 and kx>=0) after fftshift",
        "ordering": "radius^2 asc, angle asc",
        "per_freq_token_order": "Re(R), Re(G), Re(B), Im(R), Im(G), Im(B)",
        "component_dtype": "<f4",
        "byte_endianness": "little",
        "storage_dtype": "uint16",
        "bos_id": BOS_ID,
        "has_bos": True,
        "kept_freqs": K,
        "bytes_per_freq": bytes_per_freq,
        "tokens_per_row": tokens_per_row,
        "n_train_images": int(train_mat.shape[0]),
        "n_val_images": int(val_mat.shape[0]),
        "train_file": "train.npy",
        "val_file": "val.npy",
        "note": "Rows are BOS=256 followed by bytewise LE float32 of FFT coeffs in spiral order, dropping Hermitian redundancies.",
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved {out_dir / 'train.npy'} {train_mat.shape} {train_mat.dtype}")
    print(f"Saved {out_dir / 'val.npy'}   {val_mat.shape}   {val_mat.dtype}")
    print(f"Saved {out_dir / 'meta.pkl'}")
    print("Done.")

if __name__ == "__main__":
    main()
