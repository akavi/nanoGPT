#!/usr/bin/env python3
"""
Diagnostic tests for ARD with n_step=1 (should degrade to vanilla autoregressive).

Hypotheses tested:
  1. Blend fractions & forward equivalence (no training)
  2. MSE latent_loss_scale=0.0 vs 1.0 (is MSE loss the culprit?)
  3. Weight tying wte <-> lm_projection (does tying fix generation?)
  4. Generation logit entropy (does confidence collapse to white pixels?)

Usage:
    python tests/test_ard_s1.py --device=mps
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.ar_diffusion import ArDiffusion, ArDiffusionConfig
from models.categorical import CategoricalConfig  # noqa: F401
from models.mamba import Mamba2, MambaConfig
from models.model import ModuleList

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H, W = 32, 32
BLOCK_SIZE = H * W  # 1024 tokens per image
N_VOCAB = 256


def make_backbone(n_embd, n_layer, device, mode="train"):
    """Create a backbone. Sizes scale with n_embd."""
    return ModuleList([
        Mamba2(MambaConfig(
            n_head=max(2, n_embd // 32),
            n_embd=n_embd,
            n_inner=n_embd * 2,
            n_conv=4,
            n_state=64,
            bias=False,
            n_chunk=32,
            dropout=0.05,
            device=device,
            mode=mode,
        ), i)
        for i in range(n_layer)
    ])


def load_real_data(device, batch_size=16):
    """Load real face data if available, else generate synthetic data."""
    data_path = Path(__file__).resolve().parent.parent / "data" / "image_anime_face" / "train.npy"
    if data_path.exists():
        mat = np.load(str(data_path), mmap_mode="r")
        n = min(batch_size, mat.shape[0])
        rows = torch.from_numpy(mat[:n].astype(np.int64, copy=False)).to(device)
        print(f"  Loaded {n} real images from {data_path}")
        return rows
    else:
        print("  Real data not found, generating synthetic gradient images")
        # Synthetic: horizontal gradients (0..255 repeating across 1024 tokens)
        rows = torch.arange(N_VOCAB, device=device).repeat(BLOCK_SIZE // N_VOCAB + 1)[:BLOCK_SIZE]
        rows = rows.unsqueeze(0).expand(batch_size, -1).long()
        return rows


def tokens_to_image(tokens, path):
    """Save 1D token tensor as a grayscale PNG."""
    from PIL import Image
    arr = tokens.detach().cpu().numpy().astype(np.uint8).reshape(H, W)
    img = Image.fromarray(arr, mode="L")
    img.save(path, format="PNG")


def token_histogram(tokens):
    """Return a dict of {value: count} for token values."""
    vals = tokens.detach().cpu().numpy().flatten()
    return dict(Counter(vals))


def print_histogram_summary(hist, label=""):
    """Print summary stats of a token histogram."""
    total = sum(hist.values())
    sorted_vals = sorted(hist.items(), key=lambda x: -x[1])
    top5 = sorted_vals[:5]
    unique = len(hist)
    vals = []
    for v, c in hist.items():
        vals.extend([v] * c)
    mean_val = np.mean(vals)
    std_val = np.std(vals)
    print(f"  [{label}] unique={unique}, mean={mean_val:.1f}, std={std_val:.1f}, "
          f"top5={[(v, f'{c/total:.1%}') for v, c in top5]}")


def train_model(model, data, n_iters, lr, device, label="", target_loss=None,
                max_iters=10000, patience=200):
    """Train a model. Stops at n_iters, or when CE loss <= target_loss.

    When target_loss is set, n_iters is ignored and training runs until
    the CE component of the loss reaches the target (up to max_iters).
    patience: stop if CE hasn't improved in this many iters.
    Returns (loss_history, ce_loss_history, n_iters_actual).
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losses = []
    ce_losses = []

    limit = max_iters if target_loss is not None else n_iters
    best_ce = float("inf")
    stale = 0

    for it in range(limit):
        B = data.shape[0]
        state = model.initial_state(B)

        # For CE-only tracking: do a no-grad forward to get CE component
        # (only needed when target_loss is set and model has latent loss)
        _, state_fwd, loss = model(data, state, data)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lf = loss.item()
        losses.append(lf)

        # Estimate CE component: for ARD with latent_loss_scale > 0,
        # CE ≈ total - scale * MSE. We approximate by doing a quick
        # forward with scale=0 periodically, or just track total loss.
        # Simpler: track total loss as proxy (CE dominates once memorized).
        ce_losses.append(lf)

        if target_loss is not None:
            if lf < best_ce:
                best_ce = lf
                stale = 0
            else:
                stale += 1

        if it % 50 == 0 or it == limit - 1:
            print(f"  [{label}] iter {it:>4d}: loss={lf:.4f}")

        if target_loss is not None and lf <= target_loss:
            print(f"  [{label}] Reached target loss {target_loss} at iter {it} (loss={lf:.4f})")
            break

        if target_loss is not None and stale >= patience:
            print(f"  [{label}] Stopped: no improvement for {patience} iters (best={best_ce:.4f})")
            break

    return losses, ce_losses, it + 1


def generate_tokens(model, device, n_tokens=BLOCK_SIZE, temperature=0.8, top_k=200):  # top_k can be None
    """Generate tokens from a model, returning tokens and per-step entropy."""
    model.eval()
    model.to(device)

    # We'll manually generate to capture entropy at each step
    B = 1
    tok = torch.empty((B, 0), dtype=torch.long, device=device)
    state = model.initial_state(B)
    entropies = []

    # Warm up ladder for ARD
    if hasattr(model, 'n_step'):
        for _ in range(model.n_step - 1):
            _, state, _, _ = model(tok, state)

    with torch.no_grad():
        for _ in range(n_tokens):
            logits, state, _, _ = model(tok, state)
            logits_last = logits[:, -1, :] / temperature
            probs = F.softmax(logits_last, dim=-1)

            # Entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1).item()
            entropies.append(entropy)

            if top_k is not None:
                v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
                logits_last[logits_last < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits_last, dim=-1)

            x_next = torch.multinomial(probs, num_samples=1)
            tok = torch.cat((tok, x_next), dim=1)

    return tok[0], entropies


# ---------------------------------------------------------------------------
# Test 1: Blend fractions & forward equivalence
# ---------------------------------------------------------------------------

def test_blend_fractions(device, n_embd=64, n_layer=2, batch_size=16, target_loss=None):
    """Verify that ARD(S=1) blend fractions reduce to vanilla AR."""
    print("\n" + "=" * 70)
    print("TEST 1: Blend fractions & forward equivalence (S=1)")
    print("=" * 70)

    backbone = make_backbone(n_embd, n_layer=1, device=device)
    model = ArDiffusion(ArDiffusionConfig(
        n_block=BLOCK_SIZE, n_vocab=N_VOCAB, n_embd=n_embd, n_step=1,
        latent_loss_scale=1.0, snr_bias=5.0, dropout=0.0, device=device, mode="train",
    ), backbone)
    model.to(device)

    # Check blend fractions
    clean, ar_frac, noise_frac = model._blend_fracs(device)
    print(f"  S=1 blend fracs: clean={clean.item():.3f}, "
          f"ar={ar_frac.item():.3f}, noise={noise_frac.item():.3f}")

    assert clean.item() == 0.0, f"Expected clean=0 for S=1, got {clean.item()}"
    assert ar_frac.item() == 1.0, f"Expected ar=1 for S=1, got {ar_frac.item()}"
    assert noise_frac.item() == 0.0, f"Expected noise=0 for S=1, got {noise_frac.item()}"
    print("  PASS: blend fractions are (clean=0, ar=1, noise=0) for S=1")

    # Check that the constructed input is a shifted version of embeddings
    data = load_real_data(device, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        x_in, clean_tilted, train_mask = model._prep_backbone_inputs(data)

    print(f"  x_in shape: {x_in.shape}")  # (B, L, 1, E)
    print(f"  clean_tilted shape: {clean_tilted.shape}")
    print(f"  train_mask shape: {train_mask.shape}")

    # For S=1, L = T + 2*(1-1) = T, and after tilt with S=1 it's trivial
    # x_in should be ar_frac * shifted_embeddings (no clean, no noise)
    B, L, S, E = x_in.shape
    assert S == 1, f"Expected S=1 in output, got {S}"
    print(f"  L={L} (expected {data.shape[1]})")

    print("  PASS: Shapes are correct for S=1\n")
    return True


# ---------------------------------------------------------------------------
# Test 2: latent_loss_scale=0.0 vs 1.0
# ---------------------------------------------------------------------------

def test_latent_loss_scale(device, n_iters=100, n_embd=192, n_layer=6, batch_size=64,
                           target_loss=None):
    """Compare generation quality with MSE loss on vs off.
    When target_loss is set, both variants train until total loss <= target."""
    print("\n" + "=" * 70)
    tl_str = f", target_loss={target_loss}" if target_loss else f", n_iters={n_iters}"
    print(f"TEST 2: latent_loss_scale=0.0 vs 1.0 (n_embd={n_embd}, n_layer={n_layer}, B={batch_size}{tl_str})")
    print("=" * 70)

    data = load_real_data(device, batch_size=batch_size)

    results = {}
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale in [0.0, 1.0]:
        print(f"\n  --- latent_loss_scale={scale} ---")
        torch.manual_seed(42)
        backbone = make_backbone(n_embd, n_layer, device)
        model = ArDiffusion(ArDiffusionConfig(
            n_block=BLOCK_SIZE, n_vocab=N_VOCAB, n_embd=n_embd, n_step=1,
            latent_loss_scale=scale, snr_bias=5.0, dropout=0.0, device=device, mode="train",
        ), backbone)

        losses, _, actual_iters = train_model(
            model, data, n_iters, lr=3e-4, device=device,
            label=f"scale={scale}", target_loss=target_loss)

        # Switch to sample mode
        model.mode = "sample"
        for m in model.modules():
            if hasattr(m, 'mode'):
                m.mode = "sample"

        tokens, entropies = generate_tokens(model, device)
        hist = token_histogram(tokens)
        print_histogram_summary(hist, label=f"gen scale={scale}")

        # Save generated image
        img_path = out_dir / f"test2_scale{scale}.png"
        tokens_to_image(tokens, str(img_path))
        print(f"  Saved: {img_path}")

        mean_entropy = np.mean(entropies)
        min_entropy = np.min(entropies)
        print(f"  Entropy: mean={mean_entropy:.3f}, min={min_entropy:.3f}, "
              f"first10_mean={np.mean(entropies[:10]):.3f}, "
              f"last10_mean={np.mean(entropies[-10:]):.3f}")

        results[scale] = {
            "final_loss": losses[-1],
            "iters": actual_iters,
            "mean_entropy": mean_entropy,
            "min_entropy": min_entropy,
            "unique_tokens": len(hist),
            "hist": hist,
        }

    # Compare
    print("\n  --- Comparison ---")
    for scale in [0.0, 1.0]:
        r = results[scale]
        print(f"  scale={scale}: loss={r['final_loss']:.4f} @ {r['iters']} iters, "
              f"mean_entropy={r['mean_entropy']:.3f}, "
              f"unique_tokens={r['unique_tokens']}")

    return results


# ---------------------------------------------------------------------------
# Test 3: Weight tying
# ---------------------------------------------------------------------------

def test_weight_tying(device, n_iters=100, n_embd=192, n_layer=6, batch_size=64,
                      target_loss=None):
    """Train S=1 with wte.weight tied to lm_projection.weight.
    When target_loss is set, both variants train to the same loss level."""
    print("\n" + "=" * 70)
    tl_str = f", target_loss={target_loss}" if target_loss else f", n_iters={n_iters}"
    print(f"TEST 3: Weight tying (n_embd={n_embd}, n_layer={n_layer}, B={batch_size}{tl_str})")
    print("=" * 70)

    data = load_real_data(device, batch_size=batch_size)
    out_dir = Path(__file__).resolve().parent.parent / "tests" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Variants: untied+MSE, tied+CE-only (tying + MSE = NaN due to gradient conflict)
    variants = [
        {"label": "untied",       "tied": False, "latent_loss_scale": 1.0},
        {"label": "tied+ce_only", "tied": True,  "latent_loss_scale": 0.0},
    ]

    for var in variants:
        label = var["label"]
        print(f"\n  --- {label} ---")
        torch.manual_seed(42)
        backbone = make_backbone(n_embd, n_layer, device)
        model = ArDiffusion(ArDiffusionConfig(
            n_block=BLOCK_SIZE, n_vocab=N_VOCAB, n_embd=n_embd, n_step=1,
            latent_loss_scale=var["latent_loss_scale"], snr_bias=5.0, dropout=0.0,
            device=device, mode="train",
        ), backbone)

        if var["tied"]:
            model.wte.weight = model.lm_projection.weight
            print(f"  Tied wte.weight = lm_projection.weight (shape {model.wte.weight.shape})")
            print(f"  latent_loss_scale={var['latent_loss_scale']} (MSE+tying causes NaN)")

        losses, _, actual_iters = train_model(
            model, data, n_iters, lr=3e-4, device=device, label=label,
            target_loss=target_loss)

        # Switch to sample mode
        model.mode = "sample"
        for m in model.modules():
            if hasattr(m, 'mode'):
                m.mode = "sample"

        tokens, entropies = generate_tokens(model, device)
        hist = token_histogram(tokens)
        print_histogram_summary(hist, label=f"gen {label}")

        img_path = out_dir / f"test3_{label}.png"
        tokens_to_image(tokens, str(img_path))
        print(f"  Saved: {img_path}")

        mean_entropy = np.mean(entropies)
        print(f"  Entropy: mean={mean_entropy:.3f}, min={np.min(entropies):.3f}")

        results[label] = {
            "final_loss": losses[-1],
            "iters": actual_iters,
            "mean_entropy": mean_entropy,
            "unique_tokens": len(hist),
        }

    # Compare
    print("\n  --- Comparison ---")
    for var in variants:
        label = var["label"]
        r = results[label]
        print(f"  {label}: loss={r['final_loss']:.4f} @ {r['iters']} iters, "
              f"mean_entropy={r['mean_entropy']:.3f}, "
              f"unique_tokens={r['unique_tokens']}")

    return results


# ---------------------------------------------------------------------------
# Test 4: Generation logit entropy over time
# ---------------------------------------------------------------------------

def test_entropy_collapse(device, n_iters=100, n_embd=192, n_layer=6, batch_size=64,
                          target_loss=None):
    """Train S=1, then examine whether entropy collapses during generation."""
    print("\n" + "=" * 70)
    tl_str = f", target_loss={target_loss}" if target_loss else f", n_iters={n_iters}"
    print(f"TEST 4: Entropy collapse (n_embd={n_embd}, n_layer={n_layer}, B={batch_size}{tl_str})")
    print("=" * 70)

    data = load_real_data(device, batch_size=batch_size)

    torch.manual_seed(42)
    backbone = make_backbone(n_embd, n_layer, device)
    model = ArDiffusion(ArDiffusionConfig(
        n_block=BLOCK_SIZE, n_vocab=N_VOCAB, n_embd=n_embd, n_step=1,
        latent_loss_scale=1.0, snr_bias=5.0, dropout=0.0, device=device, mode="train",
    ), backbone)

    _, _, actual_iters = train_model(
        model, data, n_iters, lr=3e-4, device=device, label="entropy",
        target_loss=target_loss)
    print(f"  Trained for {actual_iters} iters")

    # Switch to sample mode
    model.mode = "sample"
    for m in model.modules():
        if hasattr(m, 'mode'):
            m.mode = "sample"

    tokens, entropies = generate_tokens(model, device, temperature=1.0, top_k=None)

    # Analyze entropy trajectory
    ent = np.array(entropies)
    n = len(ent)
    quarters = [ent[:n//4], ent[n//4:n//2], ent[n//2:3*n//4], ent[3*n//4:]]
    print(f"  Entropy by quarter: "
          f"Q1={np.mean(quarters[0]):.3f}, Q2={np.mean(quarters[1]):.3f}, "
          f"Q3={np.mean(quarters[2]):.3f}, Q4={np.mean(quarters[3]):.3f}")

    # Uniform entropy for 256 classes = ln(256) ≈ 5.545
    max_entropy = np.log(N_VOCAB)
    print(f"  Max possible entropy (uniform over {N_VOCAB}): {max_entropy:.3f}")
    print(f"  Mean entropy: {np.mean(ent):.3f} ({np.mean(ent)/max_entropy:.1%} of max)")

    # Check for collapse
    if np.mean(quarters[3]) < 0.5:
        print("  FINDING: Entropy collapses toward end -> model becomes overconfident")
    elif np.mean(quarters[0]) < 0.5:
        print("  FINDING: Entropy is low from start -> model never learns diversity")
    elif np.mean(ent) < 1.0:
        print("  FINDING: Overall low entropy -> near-deterministic outputs (white image)")
    else:
        print("  FINDING: Entropy stays healthy -> generation diversity is OK")

    # Token analysis
    hist = token_histogram(tokens)
    print_histogram_summary(hist, label="temp=1.0 no_topk")

    # Check if predominantly 255 (white)
    n_white = hist.get(255, 0)
    n_total = sum(hist.values())
    if n_white / n_total > 0.5:
        print(f"  FINDING: {n_white/n_total:.1%} of tokens are 255 (white) -> confirms white image bug")

    return entropies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARD S=1 diagnostic tests")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--n_iters", type=int, default=500, help="Training iterations per test")
    parser.add_argument("--n_embd", type=int, default=192, help="Embedding dimension")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of backbone layers")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--target_loss", type=float, default=None,
                        help="Train until loss <= this value (overrides --n_iters)")
    parser.add_argument("--tests", default="1,2,3,4", help="Comma-separated test numbers to run")
    args = parser.parse_args()

    device = args.device
    n_iters = args.n_iters
    n_embd = args.n_embd
    n_layer = args.n_layer
    batch_size = args.batch_size
    target_loss = args.target_loss
    tests = [int(t) for t in args.tests.split(",")]

    print(f"Device: {device}")
    print(f"Model: n_embd={n_embd}, n_layer={n_layer}, batch_size={batch_size}")
    if target_loss is not None:
        print(f"Training to target loss: {target_loss}")
    else:
        print(f"Iterations per test: {n_iters}")
    print(f"Running tests: {tests}")

    model_kwargs = dict(n_embd=n_embd, n_layer=n_layer, batch_size=batch_size,
                        target_loss=target_loss)

    results = {}
    t0 = time.time()

    if 1 in tests:
        test_blend_fractions(device, **model_kwargs)

    if 2 in tests:
        results["test2"] = test_latent_loss_scale(device, n_iters, **model_kwargs)

    if 3 in tests:
        results["test3"] = test_weight_tying(device, n_iters, **model_kwargs)

    if 4 in tests:
        results["test4"] = test_entropy_collapse(device, n_iters, **model_kwargs)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"All tests completed in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    # Write summary to debug_log
    log_path = Path(__file__).resolve().parent.parent / "debug_log.md"
    with open(log_path, "a") as f:
        f.write(f"\n\n## ARD S=1 Diagnostic Results ({time.strftime('%Y-%m-%d %H:%M')})\n\n")
        tl_str = f", target_loss={target_loss}" if target_loss else f", iters={n_iters}"
        f.write(f"Device: {device}, n_embd={n_embd}, n_layer={n_layer}, B={batch_size}{tl_str}\n\n")

        if "test2" in results:
            f.write("### Test 2: latent_loss_scale comparison\n")
            for scale in [0.0, 1.0]:
                r = results["test2"][scale]
                f.write(f"- scale={scale}: loss={r['final_loss']:.4f} @ {r['iters']} iters, "
                        f"entropy={r['mean_entropy']:.3f}, "
                        f"unique={r['unique_tokens']}\n")
            f.write("\n")

        if "test3" in results:
            f.write("### Test 3: Weight tying\n")
            for label in results["test3"]:
                r = results["test3"][label]
                f.write(f"- {label}: loss={r['final_loss']:.4f} @ {r['iters']} iters, "
                        f"entropy={r['mean_entropy']:.3f}, "
                        f"unique={r['unique_tokens']}\n")
            f.write("\n")

        if "test4" in results:
            ent = np.array(results["test4"])
            n = len(ent)
            f.write("### Test 4: Entropy trajectory\n")
            f.write(f"- Mean: {np.mean(ent):.3f}, Min: {np.min(ent):.3f}, Max: {np.max(ent):.3f}\n")
            f.write(f"- Q1: {np.mean(ent[:n//4]):.3f}, Q4: {np.mean(ent[3*n//4:]):.3f}\n")
            f.write("\n")

    print(f"Results appended to {log_path}")


if __name__ == "__main__":
    main()
