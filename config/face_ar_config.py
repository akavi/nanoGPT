import sys
from pathlib import Path
from typing import Any

import torch

from models.categorical import CategoricalConfig, Categorical
from models.mamba import Mamba2, MambaConfig
from train import train, TrainConfig
import torch.nn as nn
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_sampled_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
overridable = override(sys.argv, {
    "out_dir": "out-shakespeare",
    "dataset": "shakespeare_char",
    "mode": "from_scratch",  
    "device": "cuda",
    "seed":1337,
    "learning_rate":3e-4,
    "min_lr":3e-5,
    "n_layer": 10,
    "n_embd":384,
    "bias": True,
    "block_size": 1024,
})

# -----------------------------------------------------------------------------#
# Metadata / model init
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_data(overridable['dataset'], prepare_image_anime_face)
backbone = nn.ModuleList([
    Mamba2(MambaConfig(
        n_head=8,
        n_embd=overridable['n_embd'],
        n_inner=768,
        n_conv=4,
        n_state=64,

        bias=False,
        n_chunk=32,
        dropout=0.05,
        device=overridable['device'],
        mode="train" if overridable["mode"] in ["from_scratch", "resume"] else "sample",
    ), i)
for i in range(overridable['n_layer'])])
model = Categorical(CategoricalConfig(
    n_block=overridable['block_size'],
    n_vocab=meta['vocab_size'],
    n_embd=overridable['n_embd'],
    bias=overridable['bias'],
    dropout=0.05,
), backbone)

optimizer_config = OptimizerConfig(
    weight_decay=1e-1,
    learning_rate=overridable['learning_rate'],
    betas=(0.9, 0.95),
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
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    device=overridable['device'],

    decay_lr=True,
    lr_decay_iters=300,
    learning_rate=overridable['learning_rate'],
    min_lr=overridable['min_lr'],

    grad_clip=1.0,
    gradient_accumulation_steps=1,
    batch_size=256,                # also used in get_batch

    eval_only=False,
    eval_interval=250,
    eval_iters=20,
    warmup_iters=300,
    max_iters=3000,

    log_interval=1,
    always_save_checkpoint=True,
    out_dir=overridable['out_dir'],
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

    def detokenize(tokens: np.ndarray) -> Image.Image:
        """
        Inverse of the linear row-major rasterization (grayscale).
        tokens: length TOKENS_LINEAR array-like of ints in [0,255], NO BOS at front.
        Returns PIL.Image (mode 'L') of shape HxW.
        """
        t = np.asarray(tokens[1:].cpu(), dtype=np.uint8)
        if t.ndim != 1 or t.size != TOKENS_LINEAR:
            raise ValueError(f"expected 1D length {TOKENS_LINEAR}, got shape {t.shape}")
        img = t.reshape(H, W)  # row-major single channel
        img = Image.fromarray(img, mode="L")

        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

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
