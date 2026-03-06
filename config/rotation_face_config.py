import sys
import os

import torch

from models.rotation import make_rotation_backbone, RotationAttention
from models.categorical import CategoricalConfig, Categorical
from train import train, TrainConfig
from sample import sample, SampleConfig
from utils import (
    OptimizerConfig,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face
from data.image_anime_face.utils import make_get_batch, init_gen, save_image

overridable = override(sys.argv, {
    "out_dir": "out-rotation-face",
    "dataset": "image_anime_face",
    "mode": "from_scratch",
    "device": "cuda",
    "seed": 1337,
    "learning_rate": 3e-4,
    "min_lr": 3e-5,
    "n_layer": 10,
    "n_embd": 384,
    "n_attn_heads": 12,
    "d_k": 32,
    "bias": False,
    "block_size": 1024,
    "max_iters": 3000,
})

# -----------------------------------------------------------------------------#
# Init Model
# -----------------------------------------------------------------------------#

torch.manual_seed(overridable['seed'])
meta = init_sampled_data(overridable['dataset'], prepare_image_anime_face)

backbone = make_rotation_backbone(
    n_layer=overridable['n_layer'],
    n_embd=overridable['n_embd'],
    n_attn_heads=overridable['n_attn_heads'],
    d_k=overridable['d_k'],
    block_size=overridable['block_size'],
    bias=overridable['bias'],
    dropout=0.05,
)

model = Categorical(CategoricalConfig(
    n_block=overridable['block_size'],
    n_vocab=meta['vocab_size'],
    n_embd=overridable['n_embd'],
    bias=overridable['bias'],
    dropout=0.05,
), backbone)

# Re-init rotation value projections near zero (Categorical._init_weights
# overwrites with std=0.02; we need ~identity rotations at init)
for m in model.modules():
    if isinstance(m, RotationAttention):
        torch.nn.init.normal_(m.W_V.weight, std=1e-4)

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
    optimizer = configure_optimizers(model, optimizer_config)
elif mode == "resume" or mode == "sample":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        overridable['out_dir'], overridable['device'], model, optimizer_config,
    )
    print("Best val loss: ", best_val_loss)
else:
    raise ValueError(f"Unsupported mode={mode}")

get_batch = make_get_batch(overridable['dataset'], overridable['device'], overridable['block_size'])

# -----------------------------------------------------------------------------#
# TrainConfig and train() call
# -----------------------------------------------------------------------------#

train_config = TrainConfig(
    initial_iter_num=iter_num,
    initial_val_loss=best_val_loss,
    device=overridable['device'],

    decay_lr=True,
    lr_decay_iters=3000,
    learning_rate=overridable['learning_rate'],
    min_lr=overridable['min_lr'],

    grad_clip=1.0,
    gradient_accumulation_steps=1,
    batch_size=128,

    eval_only=False,
    eval_interval=250,
    eval_iters=20,
    warmup_iters=300,
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
            overridable['out_dir'], it, val_loss, cfg, mdl, opt,
        ),
        config=train_config,
    )
else:
    out_dir = overridable['out_dir']
    sample_config = SampleConfig(
        num_samples=10,
        max_new_tokens=1024,
        temperature=0.8,
        top_k=200,
        seed=1337,
        device=overridable['device'],
        compile=False,
    )
    os.makedirs(out_dir, exist_ok=True)
    sample(
        model=model,
        init_gen=init_gen,
        detokenize=lambda tokens, path: save_image(tokens, path, out_dir),
        config=sample_config,
    )
