# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-image-raster'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 1

dataset = 'image_raster'
block_size = 1024
gradient_accumulation_steps = 1
batch_size = 256 # rows/step

block_type = "mamba"
n_layer = 10 # Mamba2 block count
n_head = 8 # SSM “heads” (not attention); with headdim=64 → d_ssm=512
n_embd = 384 # d_model
n_inner = 768 # if you have an MLP head; otherwise ignore
n_conv = 4 # conv kernel (selective conv) per block
n_state = 64 # SSM state size per head
n_chunk = 32 # scan chunk/unroll (kernel perf knob)

dropout = 0.05 # or 0.0 if you use even light aug
min_lr = 3e-5
learning_rate = 3e-4 # AdamW
lr_decay_iters = 300
max_iters = 3000 # ≈ one pass compute-optimal
beta2 = 0.95

warmup_iters = 30 # ~10%
