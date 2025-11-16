import os
from pathlib import Path
from typing import Any

import torch

from model import GPTConfig, GPT
from train import train, TrainConfig  # refactored loop
from utils import (
    DataConfig,
    get_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_data,
    load_checkpoint,
    configure_optimizers,
)
from data.shakespeare_char.prepare import prepare as prepare_shakespeare

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out-shakespeare"
data_path = "shakespeare_char"
mode = "from_scratch"  # 'scratch' or 'resume'
device = "mps"
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
data_dir = os.path.join("data", data_path)
meta = init_data(data_dir, prepare_shakespeare)

# model init args
# --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
block_size = 64
model_args: dict[str, Any] = dict(
    n_state=[None, None, None, None],
    n_head=4,
    n_embd=128,
    block_size=block_size,
    bias=False,
    vocab_size=meta['vocab_size'],
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

if mode == "from_scratch":
    print("Initializing a new model from scratch")
    iter_num = 0
    best_val_loss = 1e9

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    optimizer = configure_optimizers(
        model,
        weight_decay,
        learning_rate,
        (beta1, beta2),
        "cuda" if "cuda" in device else "cpu",
    )

elif mode == "resume":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        out_dir, device, model_args
    )
elif mode == "sample":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        out_dir, device, model_args
    )
else:
    raise ValueError(f"Unsupported mode={mode}")

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

# --device=cpu
# --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 
train_config = TrainConfig(
    seed=1337,
    learning_rate=learning_rate,          # used also in optimizer
    decay_lr=True,
    warmup_iters=2000,
    lr_decay_iters=2000,
    min_lr=6e-5,
    grad_clip=1.0,
    max_iters=2000,
    gradient_accumulation_steps=5 * 8,
    batch_size=12,                # also used in get_batch
    eval_only=False,
    eval_interval=2000,
    eval_iters=20,
    log_interval=1,
    always_save_checkpoint=True,
    device=device,
    out_dir=out_dir,
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    compile=False,
)

train(
    model=model,
    optimizer=optimizer,
    get_batch=lambda split, bsz: get_config_batch(
        split,
        bsz,
        DataConfig(
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
