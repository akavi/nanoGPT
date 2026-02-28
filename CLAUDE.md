# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

nanoGPT — a minimal transformer training framework with pluggable model architectures (GPT, Mamba2 SSM, autoregressive diffusion, categorical). Includes remote GPU pod orchestration via Prime Intellect (`pi.py`).

## Common Commands

```bash
# Prepare data (run once per dataset)
uv run data/shakespeare_char/prepare.py

# Train locally
uv run train.py config/train_shakespeare_char.py

# Train with param overrides
uv run train.py config/train_shakespeare_char.py --batch_size=32 --learning_rate=1e-4

# Multi-GPU training
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Sample from trained model
uv run sample.py --out_dir=out-shakespeare-char

# Remote GPU training (Prime Intellect)
uv run pi.py up                              # provision pod
uv run pi.py train config/some_config.py     # queue training run
uv run pi.py sample --run=1                  # sample from run
uv run pi.py fetch --run=1                   # rsync outputs locally
uv run pi.py down                            # terminate pod

# Run tests
uv run -m pytest tests/
```

## Architecture

### Training Flow
Config file → model instantiation → `train.py:train()` loop → checkpoint saving. Sampling uses `sample.py` with the same config/checkpoint.

### Protocol-Based Model System
`train.py` defines a `TrainModel` protocol. Any model must implement:
- `model(x, state, y) -> (logits, new_state, loss)` — forward pass with explicit state threading
- `model.initial_state(batch_size)` — create initial state (enables RNN/SSM inference caching)
- `model.estimate_mfu()` / `model.flops_per_fwdbwd()` — performance tracking

Models live in `models/`: `model.py` (GPT transformer), `mamba.py` (Mamba2 SSM), `ar_diffusion.py`, `categorical.py` (wrapper for swappable backbones), `layer_norm.py`.

### Configuration System (`configurator.py`)
Configs are plain Python files executed via `exec()`. Command-line `--key=value` overrides are applied after. No YAML/TOML — configs directly instantiate models and set hyperparameters. Image/experimental configs in `config/` use dataclasses + `override()` helper for CLI overrides.

### Data Loading (`utils.py`)
Datasets stored as `train.bin`/`val.bin` (uint16 numpy memmap). `get_batch()` does random-offset sampling with pinned memory + async GPU transfer. Prepare scripts in `data/*/prepare.py` tokenize raw text.

### pi.py — Remote GPU Pod Manager
CLI for Prime Intellect GPU pods. State in `~/.pi/state.json`. Commands queued to remote `~/.pi_queue` file, processed by a tmux-based runner loop. Run IDs are sequential; metadata stored in `outputs/$run_id/run.json`. Each `pi train`/`sample`/`resume` always spawns a background watcher (forked via `os.fork()`) that polls queue completion, rsyncs results, and auto-shuts down the pod. When batching multiple runs, each invocation kills the previous watcher and starts a new one covering all accumulated run IDs (tracked via `watcher_run_ids` in state). **Do NOT use `pi zombify` when batching runs** — the watcher accumulation already handles keeping the pod alive until all queued runs finish. Just call `pi train` back to back.

## Key Conventions

- `state` is explicitly threaded through model forward passes (not hidden in the model) to support both transformer KV-cache and SSM inference state
- Weight tying between token embeddings and output head is standard
- `model.mode` switches between "train" and "sample" (affects attention computation)
- Checkpoints contain: model state_dict, optimizer state_dict, iter_num, best_val_loss
- `ModuleList` (in `models/model.py`) manages per-block state init/update and FLOP tracking
- DDP + gradient accumulation supported; `torchrun` for multi-GPU

## Dependencies

Python 3.12+, managed with `uv`. Key deps: torch, numpy, transformers, tiktoken, einops, wandb.

**Always use `uv run` to invoke Python** — both for scripts (`uv run pi.py ...`) and one-off commands (`uv run python -c "..."`). Never use bare `python`.
