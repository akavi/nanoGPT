from __future__ import annotations

import math
import time
from dataclasses import dataclass
from contextlib import nullcontext, AbstractContextManager
from typing import Any, Protocol, Literal
from collections.abc import Callable

import torch
from torch import Tensor
from torch.optim import Optimizer

Split = Literal["train", "val"]
DTypeStr = Literal["float32", "bfloat16", "float16"]

class TrainModel(Protocol):
    """
    Structural protocol for the model we expect.

    Any nn.Module that:
    - has initial_state(B) -> some state
    - has estimate_mfu(batch_size, dt) -> float
    - is callable as model(x, state, y) -> (logits, new_state, loss)
    - supports .to(), .train(), .eval()
    will satisfy this.
    """

    def to(self, device: str | torch.device) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...
    def eval(self) -> Any: ...

    def initial_state(self, batch_size: int) -> Any: ...
    def estimate_mfu(self, batch_size: int, dt: float) -> float: ...

    def __call__(
        self,
        x: Tensor,
        state: Any,
        y: Tensor,
    ) -> tuple[Tensor, Any, Tensor]: ...
    # (logits, new_state, loss)

GetBatchFn = Callable[[Split, int], tuple[Tensor, Tensor]]
SaveCheckpointFn = Callable[
    [int, float, "TrainConfig", TrainModel, Optimizer],
    None,
]

@dataclass
class TrainConfig:
    # --- optimization ---
    compile: bool = True
    learning_rate: float = 3e-4
    decay_lr: bool = True
    warmup_iters: int = 2_000
    lr_decay_iters: int = 600_000
    min_lr: float = 6e-5
    grad_clip: float = 1.0
    max_iters: int = 600_000
    gradient_accumulation_steps: int = 1
    batch_size: int = 64

    # --- eval ---
    eval_only: bool = False
    eval_interval: int = 200
    eval_iters: int = 100

    log_interval: int = 1
    always_save_checkpoint: bool = False

    # --- device ---
    device: str = "cuda"

    # --- I/O ---
    out_dir: str = "out"

    # --- runtime state (for resume) ---
    initial_iter_num: int = 0
    initial_val_loss: float = float("inf")


# --- main training loop -------------------------------------------------------

def train(
    model: TrainModel,
    optimizer: Optimizer,
    get_batch: GetBatchFn,
    save_checkpoint: SaveCheckpointFn,
    config: TrainConfig,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model.to(config.device)
    if config.compile:
        model = torch.compile(model)

    if torch.cuda.is_bf16_supported():
        dtype: DTypeStr = "bfloat16"
    else:
        dtype = "float16"

    device_type = "cuda" if "cuda" in config.device else "cpu"
    ctx = make_ctx(device_type, dtype)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # training loop
    x, y = get_batch("train", config.batch_size)  # initial prefetch
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    iter_num = config.initial_iter_num
    best_val_loss = config.initial_val_loss

    while True:
        lr = get_lr(iter_num, config) if config.decay_lr else config.learning_rate
        # learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # eval + checkpoint
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, get_batch, config, ctx)
            print(
                f"step {iter_num}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    save_checkpoint(
                        iter_num,
                        best_val_loss,
                        config,
                        model,
                        optimizer,
                    )
                    print(f"saving checkpoint to {config.out_dir}")

        if iter_num == 0 and config.eval_only:
            break

        for _micro_step in range(config.gradient_accumulation_steps):
            B, T = x.shape
            state = model.initial_state(B)
            with ctx:
                _, state, loss = model(x, state, y)
                loss = loss / config.gradient_accumulation_steps # scale the loss to account for gradient accumulation

            # prefetch for next step / next iter
            x, y = get_batch("train", config.batch_size)
            scaler.scale(loss).backward()

        # gradient step
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config.log_interval == 0:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = estimate_mfu(
                    model,
                    config.batch_size * config.gradient_accumulation_steps,
                    dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, "
                f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )

        iter_num += 1
        local_iter_num += 1

        if iter_num > config.max_iters:
            break


# --- helpers -----------------------------------------------------
def get_lr(it: int, cfg: TrainConfig) -> float:
    # 1) linear warmup
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    # 2) floor at min_lr
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # 3) cosine decay
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def make_ctx(
    device_type: str,
    dtype: DTypeStr,
) -> AbstractContextManager[None]:
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    if device_type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type=device_type, dtype=ptdtype)


@torch.no_grad()
def estimate_loss(
    model: TrainModel,
    get_batch: GetBatchFn,
    config: TrainConfig,
    ctx: AbstractContextManager[None],
) -> dict[str, float]:
    """
    Simple (non-chunked) evaluation over eval_iters batches.
    """
    out: dict[str, float] = {}
    model.eval()
    for split in ("train", "val"):
        total_loss_weighted = 0.0
        total_tokens = 0

        for i in range(config.eval_iters):
            x, y = get_batch(split, config.batch_size)  # [B, T]
            B, T = x.shape

            state = model.initial_state(B)
            with ctx:
                _, state, loss = model(x, state, y)

            total_loss_weighted += loss.item() * (B * T)
            total_tokens += B * T

        out[split] = total_loss_weighted / max(1, total_tokens)

    model.train()
    return out

def estimate_mfu(model, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    flops_per_iter = model.flops_per_fwdbwd() * fwdbwd_per_iter
    # express our flops throughput as ratio of A100 bfloat16 peak flops
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
    pass
