import os
import sys
from pathlib import Path
from typing import Any

import torch

from model import GPTConfig, GPT
from train import train, TrainConfig
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.shakespeare_char.prepare import prepare as prepare_shakespeare

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
overridable = override(sys.argv, {
    "out_dir": "out-shakespeare",
    "dataset": "shakespeare_char",
    "mode": "from_scratch",  # 'scratch' or 'resume'
    "device": "mps",
    "block_size": 64,
    "learning_rate":1e-3,
    "seed":1337,
})

# -----------------------------------------------------------------------------#
# Metadata / model init
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_data(overridable['dataset'], prepare_shakespeare)
model = GPT(GPTConfig(
    n_head=4,
    n_embd=128,
    n_layer=4,
    block_size=overridable['block_size'],
    bias=False,
    vocab_size=meta['vocab_size'],
    dropout=0.0,
    block_type="attention",
    device=overridable['device'],
))

optimizer_config = OptimizerConfig(
    weight_decay=1e-1,
    learning_rate=overridable['learning_rate'],
    betas=(0.9, 0.99),
    device=overridable['device'],
)

mode = overridable['mode']
if mode == "from_scratch":
    iter_num = 0
    best_val_loss = 1e9

    optimizer = configure_optimizers(
        model,
        optimizer_config,
    )

elif mode == "resume" or mode == "sample":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        out_dir, overridable['device'], model, optimizer_config,
    )
    print("Best val loss: ", best_val_loss)
else:
    raise ValueError(f"Unsupported mode={mode}")

train_config = TrainConfig(
    learning_rate=overridable['learning_rate'],
    decay_lr=True,
    warmup_iters=100,
    lr_decay_iters=2000,
    min_lr=1e-4,
    grad_clip=1.0,
    max_iters=2000,
    gradient_accumulation_steps=1,
    batch_size=12,                # also used in get_batch
    eval_only=False,
    eval_interval=250,
    eval_iters=20,
    log_interval=1,
    always_save_checkpoint=True,
    device=overridable['device'],
    out_dir=overridable['out_dir'],
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    compile=False,
)

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

print(f"{optimizer=}")
if mode == "resume" or mode == "from_scratch":
    train(
        model=model,
        optimizer=optimizer,
        get_batch=lambda split, batch_size: get_config_batch(
            split,
            batch_size,
            DataConfig(
                block_size=overridable['block_size'],
                dataset=overridable['dataset'],
                device=overridable['device'],
            ),
        ),
        save_checkpoint=lambda it, val_loss, cfg, mdl, opt: save_config_checkpoint(
            overridable['out_dir'],
            it,
            val_loss,
            cfg,
            mdl,
            opt,
        ),
        config=train_config,
    )
else:
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    def init_gen(device):
        start_ids = encode("\n")
        return (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    def detokenize(tensor, idx):
        print(f"sample {idx}:")
        print(decode(tensor.tolist()))
        print('---------------')

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=500,
        temperature=0.8,
        top_k=200,
        seed=1337,
        device=overridable['device'],
        compile=False,
    )
    os.makedirs(overridable['out_dir'], exist_ok=True)
    sample(
        model=model,
        init_gen=init_gen,
        detokenize=detokenize,
        config=sample_config,
    )
