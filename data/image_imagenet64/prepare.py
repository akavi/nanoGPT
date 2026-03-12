"""
Prebatch ImageNet-1k-64x64 into [N, H*W*C] uint8 pixel tokens.

Dataset: benjamin-paine/imagenet-1k-64x64 (HuggingFace)
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

    dataset_name = "benjamin-paine/imagenet-1k-64x64"
    seed = 1337

    print(f"Loading dataset {dataset_name}...")
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
        m = np.empty((len(dataset), TOKENS_PER_IMAGE), dtype=np.uint8)
        for i, ex in enumerate(dataset):
            img = ex["image"]
            arr = np.array(img)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.shape[:2] != (H, W):
                from PIL import Image as PILImage
                img = img.convert("RGB").resize((W, H))
                arr = np.array(img)
            m[i] = arr.reshape(-1)
            if (i + 1) % 10000 == 0:
                print(f"  processed {i+1}/{len(dataset)}")
        return m

    print("Building train matrix...")
    train_mat = to_matrix(ds_train)
    print("Building val matrix...")
    val_mat = to_matrix(ds_val)

    meta = {"vocab_size": 256}
    return train_mat, val_mat, meta
