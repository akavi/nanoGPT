from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from train import train, TrainConfig  # existing import

@dataclass
class DataConfig:
    data_dir: str
    block_size: Optional[int]
    device: str

def get_batch(
    split: str,
    batch_size_arg: int,
    cfg: DataConfig,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        x: int64 tensor of shape [B, T]
        y: int64 tensor of shape [B, T]  (next-token targets)
    """
    mat = _load_memmap(split, cfg)  # [N, L]
    N, L = mat.shape

    # choose B row indices
    row_ix = torch.randint(low=0, high=N, size=(batch_size_arg,), device="cpu")

    # determine crop length T
    # - if block_size is None or >= L, use the full row; so T = L-1
    # - else use T = block_size (must be <= L-1)
    if (cfg.block_size is None) or (cfg.block_size >= L):
        start = 0
        T = L - 1
    else:
        start = 0
        T = int(cfg.block_size)

    # gather rows (vectorized)
    rows = mat[row_ix.numpy()]  # [B, L], numpy memmap

    x_np = rows[:, start : start + T]
    y_np = rows[:, start + 1 : start + 1 + T]

    # convert to torch int64 (token ids)
    x = torch.from_numpy(x_np.astype(np.int64, copy=False))
    y = torch.from_numpy(y_np.astype(np.int64, copy=False))

    device_type = "cuda" if "cuda" in cfg.device else "cpu"
    if device_type == "cuda":
        x = x.pin_memory().to(cfg.device, non_blocking=True)
        y = y.pin_memory().to(cfg.device, non_blocking=True)
    else:
        x = x.to(cfg.device)
        y = y.to(cfg.device)

    return x, y

def _load_memmap(split: str, cfg: DataConfig) -> np.memmap:
    """Memmap the prebatched matrix for the given split."""
    fname = "train.npy" if split == "train" else "val.npy"
    path = os.path.join(cfg.data_dir, fname)
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape} in {path}")
    return arr  # shape [N, L], integer dtype


def save_checkpoint(
    out_dir: string,
    model_args
    iter_num: int,
    best_val_loss: float,
    cfg: TrainConfig,
    model,
    opt: torch.optim.Optimizer,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt = {
        "model": model._orign_mod.state_dict(),
        "optimizer": opt.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num_ckpt,
        "best_val_loss": best_val_loss_ckpt,
        "config": config_dict,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

def load_metadata(data_dir: str):
    """Load vocab_size from data_dir/meta.pkl if present, else return None."""
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta

def load_checkpoint(
    out_dir: str,
    device: str,
    model_args: dict[str, Any],
) -> tuple[GPT, torch.optim.Optimizer, int, float]:
    """
    Load model, optimizer, iter_num, best_val_loss from a checkpoint,
    updating model_args to match the checkpoint's architecture.
    """
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    optimizer = model.configure_optimizers(
        weight_decay,
        learning_rate,
        (beta1, beta2),
        "cuda" if "cuda" in device else "cpu",
    )

    return model, optimizer, checkpoint["iter_num"], checkpoint["best_val_loss"]

