"""
Prebatch FFHQ-64x64 into [N, H*W*C] uint8 pixel tokens.

Dataset: Dmini/FFHQ-64x64 (HuggingFace)
Format: 64x64 RGB images -> [N, 12288] uint8 row-major, channels-last.

Outputs (next to this file):
- train.npy: np.uint8, shape [N_train, 12288]
- val.npy:   np.uint8, shape [N_val,   12288]
- meta.pkl:  dict
"""

import numpy as np


H = W = 64
C = 3
TOKENS_PER_IMAGE = H * W * C  # 12288


def prepare():
    from datasets import load_dataset

    dataset_name = "Dmini/FFHQ-64x64"
    seed = 1337

    print(f"Loading dataset {dataset_name}...", flush=True)
    ds = load_dataset(dataset_name)

    ds_train = ds["train"].shuffle(seed=seed)
    if "validation" in ds:
        ds_val = ds["validation"].shuffle(seed=seed)
    elif "test" in ds:
        ds_val = ds["test"].shuffle(seed=seed)
    else:
        n = len(ds["train"])
        n_train = int(n * 0.9)
        ds_train = ds["train"].shuffle(seed=seed).select(range(n_train))
        ds_val = ds["train"].shuffle(seed=seed).select(range(n_train, n))

    def to_matrix(dataset):
        def convert_image(ex):
            img = ex["image"].convert("RGB").resize((W, H))
            ex["pixels"] = np.array(img, dtype=np.uint8).reshape(-1)
            return ex
        dataset = dataset.map(convert_image, remove_columns=["image"], num_proc=4)
        dataset.set_format("numpy")
        return np.stack(dataset["pixels"]).astype(np.uint8)

    print("Building train matrix...", flush=True)
    train_mat = to_matrix(ds_train)
    print("Building val matrix...", flush=True)
    val_mat = to_matrix(ds_val)

    meta = {"vocab_size": 256}
    return train_mat, val_mat, meta
