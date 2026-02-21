#!/bin/bash
set -e

LOGIN=root@86.38.238.93
PORT=22

N_TOKENS=(1 32 1024)
N_MIX=(1 10 20)

for n_tokens in "${N_TOKENS[@]}"; do
  for n_mix in "${N_MIX[@]}"; do
    dir="out-face-diffmol-t${n_tokens}-m${n_mix}"
    echo "=== Fetching ${dir} ==="
    mkdir -p "${dir}"
    scp -r -P $PORT "${LOGIN}:~/nanoGPT/${dir}/*.png" "${dir}/"
  done
done
