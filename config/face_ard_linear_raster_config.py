import sys
from pathlib import Path
from typing import Any
from models.ar_diffusion import ArDiffusion, ArDiffusionConfig
import numpy as np
from PIL import Image
import torch
import os

from models.model import ModuleList
from models.categorical import CategoricalConfig, Categorical
from models.mamba import Mamba2, MambaConfig
from train import train, TrainConfig
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_fixed_batch,
    get_sampled_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face

overridable = override(sys.argv, {
    "out_dir": "out-face-linear-raster",
    "dataset": "image_anime_face",
    "mode": "from_scratch",  
    "device": "cuda",
    "seed":1337,
    "learning_rate":3e-4,
    "min_lr":3e-5,
    "n_layer": 10,
    "n_embd":384,
    "bias": True,
    "block_size": 1024,
    "n_step": 4,
    "n_embd": 8,
    "latent_loss_scale": 0.0,
    "max_iters": 3000,
    "gamma": 0.0,
    "snr_eps": 0.1,
    "batch_size": 1,
})

# -----------------------------------------------------------------------------#
# Init Model
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_sampled_data(overridable['dataset'], prepare_image_anime_face)
backbone = ModuleList([
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
model = ArDiffusion(ArDiffusionConfig(
    n_block=overridable['block_size'],
    n_vocab=meta['vocab_size'],
    n_embd=overridable['n_embd'],
    n_step=overridable['n_step'],
    latent_loss_scale=overridable["latent_loss_scale"],
    dropout=0.05,
    device=overridable['device'],
    mode="train" if overridable["mode"] in ["from_scratch", "resume"] else "sample",
    gamma=overridable['gamma'],
    snr_eps=overridable['snr_eps'],
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
        overridable['out_dir'], overridable['device'], model, optimizer_config,
    )
    print("Best val loss: ", best_val_loss)
else:
    raise ValueError(f"Unsupported mode={mode}")

# -----------------------------------------------------------------------------#
# Data utils
# -----------------------------------------------------------------------------#

# TODO: We should store raw images
# and perform conversions on the fly
BOS_ID = 0
H, W, C = 32, 32, 1
TOKENS_LINEAR = H * W * C
TOKENS_PER_ROW = TOKENS_LINEAR + 1

# TODO: We should store raw images
# and perform conversions on the fly
def detokenize(tokens: torch.Tensor) -> Image.Image:
    """
    Inverse of the linear row-major rasterization (grayscale).
    tokens: length TOKENS_LINEAR array-like of ints in [0,255], NO BOS at front.
    Returns PIL.Image (mode 'L') of shape HxW.
    """
    t = np.asarray(tokens[1:].cpu(), dtype=np.uint8)
    if t.ndim != 1 or t.size != TOKENS_LINEAR:
        raise ValueError(f"expected 1D length {TOKENS_LINEAR}, got shape {t.shape}")
    img = t.reshape(H, W)  # row-major single channel
    return Image.fromarray(img, mode="L")

def get_batch(split, batch_size):
    rows = get_fixed_batch(
        split,
        batch_size,
        DataConfig(
            dataset=overridable['dataset'],
            device=overridable['device'],
        ),
    )
    block_size = overridable['block_size']
    x_out = rows[:, :block_size].long()           # (B, 1024)
    y_out = rows[:, 1:block_size + 1].long().contiguous()      # (B, 1024)

    return x_out, y_out

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

train_config = TrainConfig(
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    device=overridable['device'],

    decay_lr=True,
    lr_decay_iters=overridable['max_iters'],
    learning_rate=overridable['learning_rate'],
    min_lr=overridable['min_lr'],

    grad_clip=1.0,
    gradient_accumulation_steps=1,
    batch_size=overridable['batch_size'],

    eval_only=False,
    eval_interval=250,
    eval_iters=20,
    warmup_iters=overridable['max_iters'] / 10,
    max_iters=overridable['max_iters'],

    log_interval=1,
    always_save_checkpoint=True,
    out_dir=overridable['out_dir'],
    compile=False,
)
if mode == "resume" or mode == "from_scratch":
    train(
        model=model,
        optimizer=optimizer,
        get_batch=get_batch,
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
    def init_gen(device):
        return torch.zeros((1, 1), dtype=int, device=device)

    def detokenize_and_save(tokens: np.ndarray, path: str):
        img = detokenize(tokens)
        path = os.path.join(overridable['out_dir'], str(Path(path).with_suffix(".png")))
        img.save(path, format="PNG")

    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=1024,
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
        detokenize=detokenize_and_save,
        config=sample_config,
    )

"""
with torch.no_grad():
    toks = torch.tensor([[1,2,3]], device=overridable['device'])
    x_in, _ = model._prep_backbone_inputs(toks)
    print(x_in)
    print(x_in[:, :, -1, :])
    # look at which diffusion indices have nonzero content in the cleanest step
    clean = x_in[0, :, -1, :].abs().sum(dim=-1)  # (L,)
    print("L =", x_in.shape[1], "clean-nonzero idx:", clean.nonzero().squeeze(-1).tolist())
"""
