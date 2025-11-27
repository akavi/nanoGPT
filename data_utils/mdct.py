import math
import numpy as np
from typing import Tuple, Optional

import argparse
from pathlib import Path
from PIL import Image

def _round_div_pow2_scalar(x: int, s: int) -> int:
    if s == 0:
        return x
    add = 1 << (s - 1)
    if x >= 0:
        return (x + add) >> s
    else:
        return -((-x + add) >> s)

def _lift_pair_forward_scalar(x: int, y: int, p_num: int, p_shift: int,
                              u_num: int, u_shift: int) -> Tuple[int, int]:
    t = _round_div_pow2_scalar(y * p_num, p_shift)
    x = x + t
    t = _round_div_pow2_scalar(x * u_num, u_shift)
    y = y + t
    t = _round_div_pow2_scalar(y * p_num, p_shift)
    x = x + t
    return x, y

def _lift_pair_inverse_scalar(x: int, y: int, p_num: int, p_shift: int,
                              u_num: int, u_shift: int) -> Tuple[int, int]:
    t = _round_div_pow2_scalar(y * p_num, p_shift)
    x = x - t
    t = _round_div_pow2_scalar(x * u_num, u_shift)
    y = y - t
    t = _round_div_pow2_scalar(y * p_num, p_shift)
    x = x - t
    return x, y

def _rotation_to_lifts(theta: float, shift: int = 14) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    p = math.tan(theta / 2.0)
    u = -math.sin(theta)
    p_num = int(round(p * (1 << shift)))
    u_num = int(round(u * (1 << shift)))
    return (p_num, shift), (u_num, shift)

def i2i_dct1d_forward(x: np.ndarray) -> np.ndarray:
    N = int(x.shape[0])
    if N & (N - 1): 
        raise ValueError("length must be power of two")
    y = x.astype(np.int32).copy()
    n = N
    while n >= 2:
        m = n // 2
        block = y[:n].copy()
        even = block[0:n:2].astype(np.int32)
        odd  = block[1:n:2].astype(np.int32)
        sumv = (even + odd).astype(np.int32)
        diff = (even - odd).astype(np.int32)
        half = m // 2
        for k in range(half):
            i = k
            j = m - 1 - k
            theta = (k + 0.5) * math.pi / (2.0 * m)
            (p_num, p_shift), (u_num, u_shift) = _rotation_to_lifts(theta, shift=14)
            xi, yj = int(diff[i]), int(diff[j])
            xi2, yj2 = _lift_pair_forward_scalar(xi, yj, p_num, p_shift, u_num, u_shift)
            diff[i], diff[j] = xi2, yj2
        y[:n] = np.concatenate([sumv, diff])
        n = m
    return y

def i2i_dct1d_inverse(Y: np.ndarray) -> np.ndarray:
    N = int(Y.shape[0])
    if N & (N - 1): 
        raise ValueError("length must be power of two")
    y = Y.astype(np.int32).copy()
    n = 1
    while n < N:
        m = n
        n2 = 2 * m
        block = y[:n2].copy()
        sumv = block[:m].astype(np.int32)
        diff = block[m:n2].astype(np.int32)
        half = m // 2
        for k in reversed(range(half)):
            i = k
            j = m - 1 - k
            theta = (k + 0.5) * math.pi / (2.0 * m) if m > 0 else 0.0
            (p_num, p_shift), (u_num, u_shift) = _rotation_to_lifts(theta, shift=14)
            xi, yj = int(diff[i]), int(diff[j])
            xi0, yj0 = _lift_pair_inverse_scalar(xi, yj, p_num, p_shift, u_num, u_shift)
            diff[i], diff[j] = xi0, yj0
        even = ((sumv + diff) >> 1).astype(np.int32)
        odd  = ((sumv - diff) >> 1).astype(np.int32)
        inter = np.empty((n2,), dtype=np.int32)
        inter[0:n2:2] = even
        inter[1:n2:2] = odd
        y[:n2] = inter
        n = n2
    return y

def mdct_forward(img_u8: np.ndarray) -> np.ndarray:
    if img_u8.dtype != np.uint8:
        raise TypeError(f"img_u8 must be uint8, is {img_u8.dtype}")
    if img_u8.ndim != 2 or img_u8.shape[0] != img_u8.shape[1]:
        raise ValueError("img_u8 must be square 2D array")
    N = int(img_u8.shape[0])
    if N & (N - 1):
        raise ValueError("size must be a power of two")
    if N > 64:
        raise ValueError("reference impl guarantees int16 safety only for N<=64")
    X = img_u8.astype(np.int32) - 128
    for r in range(N):
        X[r, :] = i2i_dct1d_forward(X[r, :])  
    for c in range(N):
        X[:, c] = i2i_dct1d_forward(X[:, c]) 
    max_abs = int(np.max(np.abs(X)))
    assert max_abs <= 2**31, f"coeff range {max_abs} exceeds int16; reduce N or lifting precision"
    return X.astype(np.int32)

def mdct_backward(coeffs: np.ndarray) -> np.ndarray:
    if coeffs.dtype != np.int32:
        raise TypeError("coeffs must be int32")
    if coeffs.ndim != 2 or coeffs.shape[0] != coeffs.shape[1]:
        raise ValueError("coeffs must be square 2D array")
    N = int(coeffs.shape[0])
    if N & (N - 1):
        raise ValueError("size must be a power of two")
    X = coeffs.astype(np.int32)
    for c in range(N):
        X[:, c] = i2i_dct1d_inverse(X[:, c])
    for r in range(N):
        X[r, :] = i2i_dct1d_inverse(X[r, :])
    X = X + 128
    X = np.clip(X, 0, 255)
    return X.astype(np.uint8)

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def pad_to_square_pow2(u8: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad a grayscale uint8 image to an NxN (square) canvas where N is the next power of two
    >= max(H, W). Pads with zeros (black). Returns padded array and the original (H, W).
    """
    assert u8.ndim == 2 and u8.dtype == np.uint8
    h, w = u8.shape
    N = next_pow2(max(h, w))
    pad_h = N - h
    pad_w = N - w
    # pad (top, bottom), (left, right) — keep content in the top-left for easy cropping back
    padded = np.pad(u8, pad_width=((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return padded, (h, w)
