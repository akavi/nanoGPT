import os
import pickle
from dataclasses import dataclass
from typing import Optional, Any
from ast import literal_eval
from pathlib import Path
import matplotlib.pyplot as plt 
from PIL import Image

import numpy as np
import torch
from torch import Tensor
import inspect

from train import train, TrainConfig
from models.model import GPT

@dataclass
class DataConfig:
    dataset: str
    device: str

@dataclass
class OptimizerConfig:
    weight_decay: float
    learning_rate: float
    betas: list[float]
    device: str

def _load_memmap(split: str, config: DataConfig):
    """Memmap the prebatched matrix for the given split."""
    data_dir = os.path.join("data", config.dataset)
    fname = 'train.npy' if split == 'train' else 'val.npy'
    path = os.path.join(data_dir, fname)
    # np.load with mmap avoids loading into RAM; returns an array-like memmap
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape} in {fpath}")
    return arr  # shape [N, L], integer dtype

def get_fixed_batch(
    split: str,
    batch_size: int,
    config: DataConfig,
):
    """
    Returns:
        x: int64 tensor of shape [B, T]
        y: int64 tensor of shape [B, T]  (next-token targets)
    """
    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"

    mat = _load_memmap(split, config)  # [N, L]
    rows = mat[0:batch_size, :]  # shape [B, L], still numpy memmap-backed
    rows = torch.from_numpy(rows.astype(np.int64, copy=False))

    if device_type == "cuda":
        # Pin then transfer asynchronously for better throughput
        rows = rows.pin_memory().to(device, non_blocking=True)
    else:
        rows = rows.to(device)
    return rows

def get_sampled_batch(
    split: str,
    batch_size: int,
    config: DataConfig,
):
    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"

    mat = _load_memmap(split, config)  # [N, L] with L == H*W == 1024
    N, L = mat.shape

    row_ix = torch.randint(low=0, high=N, size=(batch_size,), device="cpu")
    rows_np = mat[row_ix.numpy()]  # np.int16, shape (B, L)

    rows = torch.from_numpy(rows_np)  # still integer, 0–255
    if device_type == "cuda":
        rows = rows.pin_memory().to(device, non_blocking=True)
    else:
        rows = rows.to(device)
    return rows
    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"

    mat = _load_memmap(split, config)  # [N, L]
    N, L = mat.shape

    # choose B row indices
    row_ix = torch.randint(low=0, high=N, size=(batch_size,), device="cpu")
    rows = mat[row_ix.numpy()]  # shape [B, L], still numpy memmap-backed
    rows = torch.from_numpy(rows)

    if device_type == "cuda":
        # Pin then transfer asynchronously for better throughput
        rows = rows.pin_memory().to(device, non_blocking=True)
    else:
        rows = rows.to(device)
    return rows

def get_batch(
    split: str,
    batch_size: int,
    config: DataConfig,
) -> tuple[Tensor, Tensor]:
    block_size = config.block_size
    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"
    data_dir = os.path.join("data", config.dataset)
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)
    return x, y

def save_checkpoint(
    out_dir: str,
    iter_num: int,
    best_val_loss: float,
    config: TrainConfig,
    model: Any,
    opt: torch.optim.Optimizer,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

def init_data(dataset: str, prepare_fn):
    data_dir = Path(os.path.join("data", dataset))
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        train_mat, val_mat, meta = prepare_fn()
        np.save(data_dir / "train.npy", train_mat, allow_pickle=False)
        np.save(data_dir / "val.npy",   val_mat,   allow_pickle=False)
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    else:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    return meta

def init_sampled_data(dataset: str, prepare_fn):
    data_dir = Path(os.path.join("data", dataset))
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        train_mat, val_mat, meta = prepare_fn()
        np.save(data_dir / "train.npy", train_mat, allow_pickle=False)
        np.save(data_dir / "val.npy",   val_mat,   allow_pickle=False)
        with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    else:
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

    return meta

def load_checkpoint(
    out_dir: str,
    device: str,
    model: Any,
    optimizer_config: OptimizerConfig,
) -> tuple[GPT, torch.optim.Optimizer, int, float]:
    """
    Load model, optimizer, iter_num, best_val_loss from a checkpoint,
    updating model_args to match the checkpoint's architecture.
    """
    print(f"Loading checkpoint from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # TODO validate the checkpoint can be passed into the model

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    optimizer = configure_optimizers(
        model,
        optimizer_config,
    )
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, checkpoint["iter_num"], checkpoint["best_val_loss"]

def configure_optimizers(model, config):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and "cuda" in config.device
    extra_args = dict(fused=True) if use_fused else dict()
    print(f"using fused AdamW: {use_fused}")
    return torch.optim.AdamW(
        optim_groups,
        lr=config.learning_rate,
        betas=config.betas,
        **extra_args
    )

def override(argv, config):
    for arg in argv[1:]:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in config:
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(config[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            config[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config

def view_roundtrip(
    get_batch,
    tokenize,
    detokenize,
):
    """
    Sample a batch of rows, run through tokenize -> detokenize,
    and compare reconstructed images to the original images.
    """
    rows = get_batch()

    for i, row in enumerate(rows):
        tokens = tokenize(row)
        recon_row = detokenize(tokens)

        np_row = row.cpu().numpy().reshape(32, 32)
        np_recon_row = recon_row.cpu().numpy().reshape(32, 32)

        # Prepare for display
        orig_u8 = np_row.astype(np.uint8)
        recon_u8 = np_recon_row.astype(np.uint8)

        img = Image.fromarray(orig_u8, mode="L")
        recon_img = Image.fromarray(recon_u8, mode="L")

        _, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[0].axis("off")
        axes[0].set_title("Original")

        axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
        axes[1].axis("off")
        axes[1].set_title("Reconstruction")

        plt.tight_layout()
        plt.show()
        img = Image.fromarray(np_row, mode="L")
        recon_img = Image.fromarray(np_recon_row, mode="L")


def check_roundtrip(
    get_batch,
    tokenize,
    detokenize,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> bool:
    """
    Sample a batch of rows, run through tokenize -> detokenize,
    and compare reconstructed images to the original images.
    """
    rows = get_batch()

    all_ok = True

    for i, row in enumerate(rows):
        tokens = tokenize(row)
        recon_row = detokenize(tokens)

        np_row = row.cpu().numpy()
        np_recon_row = recon_row.cpu().numpy()

        # Compare
        diff = np.abs(np_row- np_recon_row)
        max_diff = float(diff.max())
        mse = float((diff ** 2).mean())

        ok = np.allclose(np_row, np_recon_row, atol=atol, rtol=rtol)
        print(
            f"[{i}] max_diff={max_diff:.6f}, mse={mse:.6e}, "
            f"{'OK' if ok else 'MISMATCH'}"
        )

        if not ok:
            all_ok = False

    print("=> ROUNDTRIP OK" if all_ok else "=> ROUNDTRIP MISMATCH")
    return all_ok

def plot_log_token_position_means(
    get_batch,
    tokenize,
) -> None:
    """
    Draw `num_samples` rows, tokenize, shift by 2**16 to make values positive,
    take log, then compute the average log value at each token position and plot.
    """
    rows = get_batch()
    with torch.no_grad():
        # (num_samples, TOKENS_PER_ROW)
        tokens = torch.stack([tokenize(row) for row in rows], dim=0).float()
        mean_per_pos_log = tokens.mean(dim=0).cpu().numpy()
        max_per_pos_log = tokens.max(dim=0).values.cpu().numpy()
        min_per_pos_log = tokens.min(dim=0).values.cpu().numpy()

    num_samples, N = tokens.shape
    positions = np.arange(N)

    print("Max per position:", max_per_pos_log)
    print("Min per position:", min_per_pos_log)

    # --- line of best fit for max/min, excluding first position (index 0) ---
    pos_excl0 = positions[1:]
    max_excl0 = max_per_pos_log[1:]
    min_excl0 = min_per_pos_log[1:]

    # linear fit: y = m * x + b
    max_m, max_b = np.polyfit(pos_excl0, max_excl0, 1)
    min_m, min_b = np.polyfit(pos_excl0, min_excl0, 1)

    abs_excl0 = tokens[:, 1:].abs().flatten()
    print("ABS SHAPE", abs_excl0.shape)
    abs_pos = np.tile(np.arange(N - 1), (num_samples, 1)).flatten()
    print("ABS POS SHAPE", abs_pos.shape)
    abs_m, abs_b = np.polyfit(abs_pos, abs_excl0, 1)

    print(f"Max best-fit line (excluding pos 0): y = {max_m:.6g} * x + {max_b:.6g}")
    print(f"Min best-fit line (excluding pos 0): y = {min_m:.6g} * x + {min_b:.6g}")
    print(f"Abs best-fit line (excluding pos 0): y = {abs_m:.6g} * x + {abs_b:.6g}")

    # Fitted values over all positions (you can start at 1 if you prefer)
    max_fit = max_m * positions + max_b
    min_fit = min_m * positions + min_b
    abs_fit = abs_m * abs_pos + abs_b

    # --- mean plot ---
    plt.figure(figsize=(10, 4))
    plt.plot(abs_pos, abs_excl0)
    plt.xlabel("Token position")
    plt.ylabel("Average value")
    plt.title(f"Mean token value per position over {num_samples} samples")
    plt.plot(abs_pos, abs_fit, linestyle="--", label="Best-fit (max)")
    plt.tight_layout()
    plt.show()

    # --- max plot + best-fit line ---
    plt.figure(figsize=(10, 4))
    plt.plot(positions, max_per_pos_log, label="Max per position")
    plt.plot(positions, max_fit, linestyle="--", label="Best-fit (max)")
    plt.xlabel("Token position")
    plt.ylabel("Max value")
    plt.title(f"Max token value per position over {num_samples} samples")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- min plot + best-fit line ---
    plt.figure(figsize=(10, 4))
    plt.plot(positions, min_per_pos_log, label="Min per position")
    plt.plot(positions, min_fit, linestyle="--", label="Best-fit (min)")
    plt.xlabel("Token position")
    plt.ylabel("Min value")
    plt.title(f"Min token value per position over {num_samples} samples")
    plt.legend()
    plt.tight_layout()
    plt.show()

def debug_one_image(split: str, H, W, config: DataConfig):
    mat = _load_memmap(split, config)  # [N, L]
    print("mat.shape:", mat.shape, "dtype:", mat.dtype)
    row = mat[0]                       # shape (L,)

    print("row.shape:", row.shape)
    print("row min/max:", row.min(), row.max())
    print("unique count (up to 20):", np.unique(row)[:20])

    assert row.size == H * W, f"expected 1024, got {row.size}"

    arr = row.reshape(H, W).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    plt.imshow(img, cmap="gray")
    plt.title("Direct from memmap")
    plt.axis("off")
    plt.show()


def get_raw_rows(split: str, batch_size: int, config: DataConfig) -> torch.Tensor:
    """
    Return raw grayscale rows of shape (B, H*W), dtype int16/int32.
    This is a thin wrapper over _load_memmap.
    """
    device = config.device
    device_type = "cuda" if "cuda" in device else "cpu"

    mat = _load_memmap(split, config)  # [N, L] with L == H*W == 1024
    N, L = mat.shape

    row_ix = torch.randint(low=0, high=N, size=(batch_size,), device="cpu")
    rows_np = mat[row_ix.numpy()]  # np.int16, shape (B, L)

    rows = torch.from_numpy(rows_np)  # still integer, 0–255
    if device_type == "cuda":
        rows = rows.pin_memory().to(device, non_blocking=True)
    else:
        rows = rows.to(device)
    return rows

def view_roundtrip_once(
    split: str,
    config: DataConfig,
    tokenize,
    detokenize,
):
    rows = get_raw_rows(split, batch_size=1, config=config)   # (1, 1024)
    row = rows[0]                                             # (1024,)

    tokens = tokenize(row)          # (TOKENS_LINEAR,)
    recon_row = detokenize(tokens)  # (1024,)

    # Compute reconstruction error
    orig = row.cpu().numpy().astype(np.float32).reshape(32, 32)
    recon = recon_row.cpu().numpy().reshape(32, 32).astype(np.float32)
    max_err = np.abs(orig - recon).max()
    print("max abs reconstruction error:", max_err)

    # Prepare for display
    orig_u8 = orig.clip(0, 255).astype(np.uint8)
    recon_u8 = recon.clip(0, 255).astype(np.uint8)

    img = Image.fromarray(orig_u8, mode="L")
    recon_img = Image.fromarray(recon_u8, mode="L")

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0].axis("off")
    axes[0].set_title("Original")

    axes[1].imshow(recon_img, cmap="gray", vmin=0, vmax=255)
    axes[1].axis("off")
    axes[1].set_title("Reconstruction")

    plt.tight_layout()
    plt.show()
