#!/bin/bash
# Build a composite grid: outer grid is n_tokens (cols) x n_mix (rows),
# each cell is a 3x3 grid of 9 sample images from that run.
# 3px white gutters within each cell, 8px white gutters between cells.

set -e

N_TOKENS=(1 32 1024)
N_MIX=(1 10 20)
INNER_GAP=3
OUTER_GAP=8
OUT="grid_composite.png"

# Build each run's 3x3 tile
TILE_DIR=$(mktemp -d)
trap "rm -rf $TILE_DIR" EXIT

for n_mix in "${N_MIX[@]}"; do
  for n_tokens in "${N_TOKENS[@]}"; do
    dir="out-face-diffmol-t${n_tokens}-m${n_mix}"
    tile="${TILE_DIR}/tile_t${n_tokens}_m${n_mix}.png"
    # Take images 0-8, montage into 3x3 with 3px gaps
    montage \
      "${dir}/0.png" "${dir}/1.png" "${dir}/2.png" \
      "${dir}/3.png" "${dir}/4.png" "${dir}/5.png" \
      "${dir}/6.png" "${dir}/7.png" "${dir}/8.png" \
      -tile 3x3 -geometry +${INNER_GAP}+${INNER_GAP} -background white \
      "$tile"
  done
done

# Assemble tiles into outer 3x3 grid (n_tokens across, n_mix down)
ROWS=()
for n_mix in "${N_MIX[@]}"; do
  row_tiles=()
  for n_tokens in "${N_TOKENS[@]}"; do
    row_tiles+=("${TILE_DIR}/tile_t${n_tokens}_m${n_mix}.png")
  done
  row="${TILE_DIR}/row_m${n_mix}.png"
  montage "${row_tiles[@]}" -tile 3x1 -geometry +${OUTER_GAP}+0 -background white "$row"
  ROWS+=("$row")
done

montage "${ROWS[@]}" -tile 1x3 -geometry +0+${OUTER_GAP} -background white "$OUT"
echo "Saved $OUT"
