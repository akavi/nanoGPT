import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import os

from models.model import ModuleList
from models.mamba import Mamba2, MambaConfig
from models.mol import MoLConfig, MoL
from train import train, TrainConfig
from utils import (
    OptimizerConfig,
    DataConfig,
    get_sampled_batch as get_config_batch,
    save_checkpoint as save_config_checkpoint,
    init_sampled_data,
    load_checkpoint,
    configure_optimizers,
    override,
)
from data.image_anime_face.prepare import prepare as prepare_image_anime_face

overridable = override(sys.argv, {
    "out_dir": "out-face-diffmol",
    "dataset": "image_anime_face",
    "mode": "from_scratch",
    "device": "cuda",
    "seed": 1337,
    "learning_rate": 1e-4,
    "min_lr": 3e-5,
    "n_layer": 10,
    "n_embd": 384,
    "bias": True,
    "block_size": 1024,
    "n_mix": 1,
    "n_tokens": 1,
    "max_iters": 3000,
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
        mode="train" if overridable["mode"] in ["from_scratch", "resume", "eval_tf"] else "sample",
    ), i)
for i in range(overridable['n_layer'])])
model = MoL(MoLConfig(
    n_block=overridable['block_size'],
    n_embd=overridable['n_embd'],
    n_mix=overridable["n_mix"],
    n_tokens=overridable["n_tokens"],
    bias=overridable['bias'],
    dropout=0.05,
), backbone)

optimizer_config = OptimizerConfig(
    weight_decay=1e-1,
    learning_rate=overridable['learning_rate'],
    betas=[0.9, 0.95],
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

elif mode == "resume" or mode == "sample" or mode == "eval_tf":
    model, optimizer, iter_num, best_val_loss = load_checkpoint(
        overridable['out_dir'], overridable['device'], model, optimizer_config,
    )
    print("Best val loss: ", best_val_loss)
else:
    raise ValueError(f"Unsupported mode={mode}")

# -----------------------------------------------------------------------------#
# Data utils
# -----------------------------------------------------------------------------#

H, W, C = 32, 32, 1
N_TOTAL = H * W * C
N_TOKENS = overridable['n_tokens']
N_SLICE = N_TOTAL // N_TOKENS
T = N_SLICE * N_TOKENS  # sequence length (= N_TOTAL when evenly divisible)

assert N_TOTAL % N_TOKENS == 0, f"n_total={N_TOTAL} not divisible by n_tokens={N_TOKENS}"


def make_diffusion_sequence(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the input/target pair for inner-loop-diffusion + outer-loop-autoregression.

    pixels: [B, N_TOTAL] float in [-1, 1]
    Returns x, y both [B, T, N_TOKENS].

    For each slice i (of N_TOKENS pixels), we have N_TOKENS autoregressive steps
    that progressively denoise the slice from Gaussian noise to clean.

    Step j=0 (bridge): input mixes clean *previous* slice with noise.
    Steps j>0 (denoise): input mixes clean *current* slice with noise.
    Target at every step: clean current slice (x_0 prediction).

    alpha schedule:  j=0 -> 1/N_TOKENS,  j>0 -> j/N_TOKENS
    """
    B = pixels.shape[0]
    device = pixels.device

    # [B, N_SLICE, N_TOKENS]
    clean = pixels[:, :T].view(B, N_SLICE, N_TOKENS)

    # Previous-slice signal for bridge steps (zeros = BOS for first slice)
    bos = torch.zeros(B, 1, N_TOKENS, device=device)
    clean_prev = torch.cat([bos, clean[:, :-1, :]], dim=1)  # [B, N_SLICE, N_TOKENS]

    # Alpha schedule (flat): step j -> alpha_j
    alphas = torch.arange(N_TOKENS, device=device, dtype=torch.float32) / N_TOKENS
    alphas[0] = 1.0 / N_TOKENS  # bridge step

    # Build inputs as a progressive Markov chain: [B, N_SLICE, N_TOKENS_steps, N_TOKENS_values]
    inputs = torch.empty(B, N_SLICE, N_TOKENS, N_TOKENS, device=device)

    # Cleanest denoising step (j = N_TOKENS-1): standard noising of clean
    a = alphas[-1]
    inputs[:, :, -1, :] = a * clean + (1 - a) * torch.randn_like(clean)

    # Work backwards: each noisier step is built from the next cleaner step + fresh noise
    for j in range(N_TOKENS - 2, 0, -1):
        r = alphas[j] / alphas[j + 1]
        inherited = r * (1 - alphas[j + 1])
        sigma = ((1 - alphas[j]) ** 2 - inherited ** 2).clamp(min=0).sqrt()
        inputs[:, :, j, :] = r * inputs[:, :, j + 1, :] + sigma * torch.randn_like(clean)

    # Bridge step (j=0): mixes clean_prev with noise (not part of progressive chain)
    a0 = alphas[0]
    inputs[:, :, 0, :] = a0 * clean_prev + (1 - a0) * torch.randn_like(clean)

    # Target at every step is clean (x_0 prediction)
    targets = clean.unsqueeze(2).expand_as(inputs).clone()

    # Flatten slice & step dims -> [B, T, N_TOKENS]
    x = inputs.reshape(B, T, N_TOKENS)
    y = targets.reshape(B, T, N_TOKENS)
    return x, y


def get_batch(split, batch_size):
    rows = get_config_batch(
        split,
        batch_size,
        DataConfig(
            dataset=overridable['dataset'],
            device=overridable['device'],
        ),
    )
    pixels = rows[:, :N_TOTAL].to(dtype=torch.float32) / 127.5 - 1.0  # [B, N_TOTAL]
    return make_diffusion_sequence(pixels)


def detokenize(tokens: torch.Tensor) -> Image.Image:
    """
    tokens: [T, N_TOKENS] float — the full generated sequence.
    We take the clean output (last step of each slice) and reassemble the image.
    """
    # Reshape to [N_SLICE, N_TOKENS_steps, N_TOKENS_values]
    seq = tokens.view(N_SLICE, N_TOKENS, N_TOKENS)
    # Last step of each slice is the clean prediction
    clean = seq[:, -1, :]  # [N_SLICE, N_TOKENS]
    pixels = clean.reshape(-1)  # [N_TOTAL]
    # Map from [-1, 1] to [0, 255]
    pixels = ((pixels + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
    img = pixels.reshape(H, W)
    return Image.fromarray(img, mode="L")


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
    eval_interval=100,
    eval_iters=20,
    warmup_iters=overridable['max_iters'] // 10,
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
elif mode == "sample":
    from contextlib import nullcontext
    from torch.nn import functional as F

    device = overridable['device']
    num_samples = 10
    temperature = 0.8
    block_size = overridable['block_size']

    torch.manual_seed(1337)
    model.to(device)
    model.eval()
    os.makedirs(overridable['out_dir'], exist_ok=True)

    device_type = "cuda" if "cuda" in device else "cpu"
    if device_type == "cuda" and torch.cuda.is_bf16_supported():
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    elif device_type == "cuda":
        ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    else:
        ctx = nullcontext()

    def sample_mol(ml, m, ls, temp):
        """Sample from mixture of logistics. Inputs: [..., K]."""
        pi = torch.softmax(ml, dim=-1)
        comp = torch.distributions.Categorical(pi).sample()
        oh = F.one_hot(comp, num_classes=pi.size(-1)).float()
        mu_k = (m * oh).sum(-1)
        log_s_k = (ls * oh).sum(-1)
        s_k = (F.softplus(log_s_k) + 1e-8) * temp
        u = torch.rand_like(mu_k).clamp_(1e-6, 1 - 1e-6)
        return mu_k + s_k * (torch.log(u) - torch.log1p(-u))

    # Alpha schedule matching training
    alphas = torch.arange(N_TOKENS, device=device, dtype=torch.float32) / N_TOKENS
    alphas[0] = 1.0 / N_TOKENS
    alpha_bridge = alphas[0]

    with torch.no_grad(), ctx:
        for k in range(num_samples):
            clean_prev = torch.zeros(1, N_TOKENS, device=device)  # BOS
            clean_slices = []
            seq = torch.empty(1, 0, N_TOKENS, device=device)
            state = model.initial_state(1)

            for i in range(N_SLICE):
                # Bridge step: mix clean previous slice with noise
                noise = torch.randn(1, N_TOKENS, device=device)
                bridge = alpha_bridge * clean_prev + (1 - alpha_bridge) * noise
                seq = torch.cat([seq, bridge.unsqueeze(1)], dim=1)

                for j in range(N_TOKENS):
                    y_cond = seq if seq.size(1) <= block_size else seq[:, -block_size:]
                    (ml, m, ls), state, _ = model(y_cond, state, targets=None)
                    pred_clean = sample_mol(ml[:, -1], m[:, -1], ls[:, -1], temperature)

                    if j == N_TOKENS - 1:
                        # Last step: use clean prediction directly
                        clean_prev = pred_clean
                        clean_slices.append(pred_clean.squeeze(0))
                    else:
                        # Construct next noisy input from predicted clean + schedule
                        a_next = alphas[j + 1]
                        next_input = a_next * pred_clean + (1 - a_next) * torch.randn_like(pred_clean)
                        seq = torch.cat([seq, next_input.unsqueeze(1)], dim=1)

                if (i + 1) % 50 == 0:
                    print(f"  sample {k}: slice {i+1}/{N_SLICE}")

            # Assemble image from clean slices
            pixels = torch.cat(clean_slices, dim=0)  # [N_TOTAL]
            pixels = ((pixels + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            img = Image.fromarray(pixels.reshape(H, W), mode="L")
            path = os.path.join(overridable['out_dir'], f"{k}.png")
            img.save(path, format="PNG")
            print(f"Saved sample {k}")

elif mode == "eval_tf":
    # Teacher-forced eval: run the training batch through the model and
    # visualize predictions to check if forward-pass outputs are meaningful.
    from torch.nn import functional as F

    device = overridable['device']
    model.to(device)
    model.eval()
    os.makedirs(overridable['out_dir'], exist_ok=True)

    with torch.no_grad():
        x, y = get_batch("train", 1)
        state = model.initial_state(1)
        (mix_logits, mu, log_s), _, loss = model(x, state, targets=y)
        print(f"Teacher-forced loss: {loss.item():.4f}")

        # Use the mean of the highest-weight component as the prediction
        best = mix_logits.argmax(dim=-1)                            # [B, T, N]
        oh = F.one_hot(best, num_classes=mix_logits.size(-1)).float()
        pred = (mu * oh).sum(-1)                                    # [B, T, N]

        # pred is the model's teacher-forced prediction of clean at every step
        # Reshape to [N_SLICE, N_TOKENS_steps, N_TOKENS_values]
        pred_seq = pred[0].view(N_SLICE, N_TOKENS, N_TOKENS)

        # Show sequential build-up: iterate over global sequence positions.
        # At each snapshot, completed slices use their final-step prediction,
        # the current slice uses its current step, future slices are gray.
        gray = 128
        n_snapshots = min(T, 1024)  # cap number of saved images
        step_indices = [round(i * (T - 1) / max(n_snapshots - 1, 1)) for i in range(n_snapshots)]
        # also snapshot at each slice boundary
        for s in range(1, N_SLICE):
            boundary = s * N_TOKENS - 1  # last step of previous slice
            if boundary not in step_indices and boundary < T:
                step_indices.append(boundary)
        step_indices = sorted(set(step_indices))

        for global_t in step_indices:
            cur_slice = global_t // N_TOKENS
            cur_step = global_t % N_TOKENS

            pixels = np.full(N_TOTAL, gray, dtype=np.uint8)
            # Completed slices: use final denoising step
            for s in range(cur_slice):
                vals = pred_seq[s, -1, :]  # [N_TOKENS]
                p = ((vals + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
                pixels[s * N_TOKENS : (s + 1) * N_TOKENS] = p
            # Current slice: use current denoising step
            vals = pred_seq[cur_slice, cur_step, :]
            p = ((vals + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
            pixels[cur_slice * N_TOKENS : (cur_slice + 1) * N_TOKENS] = p

            img = Image.fromarray(pixels.reshape(H, W), mode="L")
            path = os.path.join(overridable['out_dir'], f"tf_t{global_t:04d}.png")
            img.save(path, format="PNG")
            print(path)
            print(f"Saved tf_t{global_t:04d}.png (slice {cur_slice}, step {cur_step})")

        # Also save the actual clean target for comparison
        target_seq = y[0].view(N_SLICE, N_TOKENS, N_TOKENS)
        clean_target = target_seq[:, -1, :]
        pixels = clean_target.reshape(-1)
        pixels = ((pixels + 1.0) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        img = Image.fromarray(pixels.reshape(H, W), mode="L")
        path = os.path.join(overridable['out_dir'], "tf_target.png")
        img.save(path, format="PNG")
        print("Saved target image")
