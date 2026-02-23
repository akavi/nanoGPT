# ARD n_step=1 Debugging Log

## Problem Statement
- **Works**: `face_linear_raster_config.py` (Categorical) — good generations
- **Works (sort of)**: `face_ard_linear_raster_config.py` with n_step>1, latent_loss_scale=1.0 — noisy but recognizable faces
- **Broken**: `face_ard_linear_raster_config.py` with n_step=any, latent_loss_scale=0.0 — pure noise
- **Broken**: `face_ard_linear_raster_config.py` with n_step=1, latent_loss_scale=1.0 — purely white images

Surprise: ARD with n_step=1 should degrade to Categorical (modulo in_norm), but produces white images instead.

## Structural Differences: ARD(S=1) vs Categorical

| Aspect | Categorical | ARD (S=1) |
|--------|------------|-----------|
| Weight tying | `wte.weight = lm_head.weight` | Separate `wte` and `lm_projection` |
| Input norm | None | `in_norm` (SubLatentLayerNorm: zero-mean, unit-var, NO learnable params) |
| Output norm | `ln_f` (LayerNorm with learnable scale/bias) | `out_norm` (SubLatentLayerNorm, no learnable params) |
| Pos emb norm | None | Normalized to zero-mean, unit-var |
| Data pipeline | Prepends BOS=0, shifts right | No BOS, internal AR shift via blend fracs |
| Loss | CE only | CE + latent_loss_scale * MSE |
| lm_head input dim | n_embd (384) | n_embd_per_step (384 when S=1) |

## Hypothesis 1: MSE latent loss at S=1 creates a degenerate fixed point (PRIMARY)

With S=1, the blend fractions are: clean=0, ar=1, noise=0.
- Input at position j: `in_norm(wte(toks[j-1]))` — pure AR, correct
- CE target at position j: `toks[j]` — correct
- MSE target at position j: `in_norm(wte(toks[j]))` — embedding of the TARGET token

The MSE loss forces `out_norm(backbone_output[j]) ≈ in_norm(wte(toks[j]))`.

During **generation**, the feedback loop is:
1. Output y → lm_projection(y) → logits → sample token t
2. Next input = in_norm(wte(t))

If MSE has taught the model to output vectors close to `in_norm(wte(target))`,
then the model effectively learns: given `in_norm(wte(t_{prev}))`, output `in_norm(wte(t_{next}))`.

But if the model has over-memorized this mapping, it may have learned a near-identity
or fixed-point map: given embedding X, output something close to X. This would cause
the model to output the same token forever → constant (white) image.

**Why this doesn't happen with S>1**: With S>1, the MSE targets are diluted across
multiple sublatents at different noise levels. The cleanest sublatent (used for CE) is
only one of S sublatents. The MSE constraint is spread across the full ladder, making
it less likely to create a single fixed-point attractor.

**Key test**: Run ARD with S=1, latent_loss_scale=0.0. If this works like Categorical,
it confirms MSE is the culprit.

## Hypothesis 2: No weight tying breaks the autoregressive feedback loop

In Categorical: `logit_k = dot(output, wte[k])` — input and output spaces are aligned.
In ARD: `logit_k = dot(output, lm_projection.weight[k])` — separate from wte.

During generation, the model outputs y, samples token k via lm_projection, then
receives `in_norm(wte[k])` as the next input. If lm_projection and wte are misaligned,
the model's expectations about what input follows a given output are violated.

**Key test**: Add weight tying to ARD (tie wte and lm_projection) and see if S=1 works.

## Hypothesis 3: SubLatentLayerNorm (in_norm) collapses token representations

`in_norm` normalizes each embedding to zero mean, unit variance without learnable
scale/bias. Unlike standard LayerNorm, there's no affine transform to recover
expressivity. This could make different token embeddings too similar.

For a 384-dim embedding normalized to unit variance, the maximum cosine similarity
between random vectors is low. But after training, the learned wte might cluster,
and in_norm could amplify this clustering.

**Key test**: Check cosine similarity distribution of in_norm(wte) embeddings after training.
The model already has `check_embedding_collisions()` for this.

## Hypothesis 4: latent_loss_scale=0.0 fails for S>1 because unsupervised sublatents corrupt the chain

With S>1 and latent_loss_scale=0.0:
- Only CE on the cleanest sublatent (s=S-1) gets gradient
- Sublatents s=0..S-2 are unsupervised
- During generation, `shifted = prev_out[:, :, :-1, :]` feeds previous sublatent outputs
  into the next position's higher rungs
- If those sublatent outputs are garbage → corrupted input → noise output

With S=1: `shifted` is always zero (no lower rungs to shift up). So the unsupervised
sublatent issue doesn't apply. **S=1 with latent_loss_scale=0.0 should work like Categorical.**

## Test Plan

1. **test_blend_fracs.py** — Verify blend fractions and tilt for S=1 (pure math, no GPU)
2. **test_ard_s1_equivalence.py** — Compare ARD(S=1) forward pass with Categorical
3. **test_generation_logits.py** — Train small model, examine generation logit distributions
4. **test_latent_loss_effect.py** — Compare S=1 with latent_loss_scale=0 vs 1.0

## Findings

(To be filled in as tests run)


## ARD S=1 Diagnostic Results (2026-02-22 21:13)

Device: mps, iters: 100



## ARD S=1 Diagnostic Results (2026-02-22 21:16)

Device: mps, iters: 100

### Test 2: latent_loss_scale comparison
- scale=0.0: loss=4.1414, entropy=5.034, unique=247
- scale=1.0: loss=5.7769, entropy=5.147, unique=246

### Test 3: Weight tying
- untied: loss=5.7538, entropy=5.232, unique=241
- tied: loss=5.3632, entropy=5.214, unique=246

### Test 4: Entropy trajectory
- Mean: 5.362, Min: 4.754, Max: 5.542
- Q1: 5.348, Q4: 5.370



## ARD S=1 Diagnostic Results (2026-02-22 21:28)

Device: mps, iters: 500

### Test 2: latent_loss_scale comparison
- scale=0.0: loss=1.3709, entropy=3.249, unique=223
- scale=1.0: loss=2.1460, entropy=3.124, unique=231

### Test 3: Weight tying
- untied: loss=2.0349, entropy=3.012, unique=229
- tied: loss=3.4690, entropy=4.411, unique=243

### Test 4: Entropy trajectory
- Mean: 4.118, Min: 3.236, Max: 4.987
- Q1: 4.127, Q4: 4.127



## ARD S=1 Diagnostic Results (2026-02-23 07:00)

Device: cuda, n_embd=192, n_layer=6, B=64, target_loss=0.1

### Test 2: latent_loss_scale comparison
- scale=0.0: loss=0.0994 @ 388 iters, entropy=1.463, unique=230
- scale=1.0: loss=0.0997 @ 751 iters, entropy=0.885, unique=215

### Test 3: Weight tying
- untied: loss=0.0979 @ 1489 iters, entropy=0.552, unique=215
- tied+ce_only: loss=0.0990 @ 579 iters, entropy=1.947, unique=225

### Test 4: Entropy trajectory
- Mean: 1.185, Min: 0.007, Max: 4.507
- Q1: 0.176, Q4: 1.516

