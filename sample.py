from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor

@dataclass
class SampleConfig:
    out_dir: str = "out"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compile: bool = False 

InitGenFn = Callable[[str], Tensor]                  
DetokenizeFn = Callable[[Tensor, str], None]        


def sample(
    model: Any,               # e.g. GPT
    init_gen: InitGenFn,
    detokenize: DetokenizeFn,
    config: SampleConfig,
) -> None:
    """
    Run autoregressive sampling from a trained model.

    `model` is assumed to already have weights loaded.
    `init_gen(device)` should return a [1, T] int64 tensor of token ids.
    `detokenize(tokens, path)` should write / print the decoded text.
    """

    torch.manual_seed(config.seed)
    if "cuda" in config.device and torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in config.device else "cpu"

    # match the train loop behavior: bf16 if available on CUDA, else fp16 on CUDA,
    # fp32 on CPU
    if device_type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype_str = "bfloat16"
        else:
            dtype_str = "float16"
    else:
        dtype_str = "float32"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_str]

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.to(config.device)
    model.eval()
    if config.compile:
        model = torch.compile(model)

    x = init_gen(config.device)  # [1, T]
    with torch.no_grad():
        with ctx:
            for k in range(config.num_samples):
                state = model.initial_state(1)
                y = model.generate(
                    x,
                    config.max_new_tokens,
                    state,
                    temperature=config.temperature,
                    top_k=config.top_k,
                )
                detokenize(y[0], str(k))
