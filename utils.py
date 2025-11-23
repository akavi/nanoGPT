import os
import pickle
from dataclasses import dataclass
from typing import Optional, Any
from ast import literal_eval

import numpy as np
import torch
from torch import Tensor
import inspect

from train import train, TrainConfig  # existing import
from models.model import GPT

@dataclass
class DataConfig:
    dataset: str
    block_size: Optional[int]
    device: str

@dataclass
class OptimizerConfig:
    weight_decay: float
    learning_rate: float
    betas: list[float]
    device: str

def get_sampled_batch(
    split: str,
    batch_size: int,
    config: DataConfig,
) -> tuple[Tensor, Tensor]:
    """
    Returns:
        x: int64 tensor of shape [B, T]
        y: int64 tensor of shape [B, T]  (next-token targets)
    """
    mat = _load_memmap(split, "npy", config)  # [N, L]
    N, L = mat.shape

    # choose B row indices
    row_ix = torch.randint(low=0, high=N, size=(batch_size,), device="cpu")

    # determine crop length T
    # - if block_size is None or >= L, use the full row; so T = L-1
    # - else use T = block_size (must be <= L-1)
    start = 0
    T = L - 1 if (config.block_size is None) or (config.block_size >= L) else int(config.block_size)

    rows = mat[row_ix.numpy()]  # [B, L], numpy memmap
    x_np = rows[:, start : start + T]
    y_np = rows[:, start + 1 : start + 1 + T]

    x = torch.from_numpy(x_np.astype(np.int64, copy=False))
    y = torch.from_numpy(y_np.astype(np.int64, copy=False))

    device_type = "cuda" if "cuda" in config.device else "cpu"
    if device_type == "cuda":
        x = x.pin_memory().to(config.device, non_blocking=True)
        y = y.pin_memory().to(config.device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)

    return x, y

def get_batch(
    split: str,
    batch_size: int,
    config: DataConfig,
) -> tuple[Tensor, Tensor]:
    block_size = config.block_size
    device_type = "cuda" if "cuda" in config.device else "cpu"
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

def _load_memmap(split: str, suffix: str, config: DataConfig) -> np.memmap:
    data_dir = os.path.join("data", config.dataset)
    fname = f"train.{suffix}" if split == "train" else f"val.{suffix}"
    path = os.path.join(data_dir, fname)
    return np.load(path, mmap_mode="r")

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
    """Load vocab_size from data_dir/meta.pkl if present, else return None."""
    data_dir = os.path.join("data", dataset)
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        train_ids, val_ids, meta = prepare_fn()
        train_ids.tofile(os.path.join(data_dir, 'train.bin'))
        val_ids.tofile(os.path.join(data_dir, 'val.bin'))
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

