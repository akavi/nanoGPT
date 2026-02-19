#!/bin/bash
set -e

for n_tokens in 1 32 1024; do
  for n_mix in 1 10 20; do
    out_dir="out-face-diffmol-t${n_tokens}-m${n_mix}"
    echo "=== Training n_tokens=${n_tokens} n_mix=${n_mix} -> ${out_dir} ==="
    uv run python3 config/face_diffmol_config.py \
      --n_tokens="${n_tokens}" \
      --n_mix="${n_mix}" \
      --out_dir="${out_dir}"

    echo "=== Sampling n_tokens=${n_tokens} n_mix=${n_mix} -> ${out_dir} ==="
    uv run python3 config/face_diffmol_config.py \
      --n_tokens="${n_tokens}" \
      --n_mix="${n_mix}" \
      --out_dir="${out_dir}" \
      --mode=sample
  done
done
