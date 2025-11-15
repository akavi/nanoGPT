"""
Single-GPU training script 
"""

import os
from pathlib import Path
from typing import Any

import torch

from model import GPTConfig, GPT
from train import train, TrainConfig  # refactored loop
from utils import (
    GetBatchConfig,
    get_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    load_metadata,
    load_checkpoint,
)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
init_from = "scratch"  # 'scratch' or 'resume'
device = "cuda" if torch.cuda.is_available() else "cpu"
# data
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config_dict = {k: globals()[k] for k in config_keys}  # useful for logging
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------#
# Metadata / model init
# -----------------------------------------------------------------------------#

base_dir = Path(os.path.dirname(__file__))
data_dir = os.path.join("data", "openwebtext")
meta = load_metadata(data_dir)

# model init args
block_size = 1024
model_args: dict[str, Any] = dict(
    n_layer=12,
    n_head=12,
    n_embd=768,
    n_inner=1536,      # 2 * n_embd
    n_state=384,
    n_chunk=64,
    block_size=block_size,
    bias=False,
    vocab_size=None,   # filled below
    dropout=0.0,
    block_type="attention",
    device=device,
)

# adamw optimizer
learning_rate = 6e-4     # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# -----------------------------------------------------------------------------

if init_from == "scratch":
    print("Initializing a new model from scratch")
    iter_num = 0
    best_val_loss = 1e9

    model_args["vocab_size"] = meta["vocab_size"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    optimizer = model.configure_optimizers(
        weight_decay,
        learning_rate,
        (beta1, beta2),
        "cuda" if "cuda" in device else "cpu",
    )

elif init_from == "resume":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        out_dir, device, model_args
    )
else:
    raise ValueError(f"Unsupported init_from={init_from!r}")

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

train_config = TrainConfig(
    seed=1337,
    learning_rate=learning_rate,          # used also in optimizer
    decay_lr=True,
    warmup_iters=2000,
    lr_decay_iters=600_000,
    min_lr=6e-5,
    grad_clip=1.0,
    max_iters=600_000,
    gradient_accumulation_steps=5 * 8,
    batch_size=12,                # also used in get_batch
    eval_only=False,
    eval_interval=2000,
    eval_iters=200,
    log_interval=1,
    always_save_checkpoint=True,
    device=device,
    out_dir=out_dir,
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
)

train(
    model=model,
    optimizer=optimizer,
    get_batch=lambda split, bsz: get_config_batch(
        split,
        bsz,
        GetBatchConfig(
            block_size=block_size,
            data_dir=data_dir,
            device=device,
        ),
    ),
    save_checkpoint=lambda it, val_loss, cfg, mdl, opt: save_config_checkpoint(
        out_dir,
        it,
        val_loss,
        cfg,
        mdl,
        opt,
        model_args=model_args,
        config_dict=config_dict,
    ),
    config=train_config,
)
