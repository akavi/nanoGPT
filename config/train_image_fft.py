# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-image-fft'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'image-fft'
wandb_run_name = 'image-fft'

dataset = 'image_fft'
block_size = 2816
gradient_accumulation_steps = 1
batch_size = 32 # rows/step

block_type = "mamba"
n_layer = 12 # Mamba2 block count
n_head = 8 # SSM “heads” (not attention); with headdim=64 → d_ssm=512
n_embd = 256 # d_model
n_inner = 512 # if you have an MLP head; otherwise ignore
n_conv = 4 # conv kernel (selective conv) per block
n_state = 64 # SSM state size per head
n_chunk = 256 # scan chunk/unroll (kernel perf knob)

dropout = 0.05 # or 0.0 if you use even light aug
learning_rate = 3e-4 # AdamW
max_iters = 3000 # ≈ one pass compute-optimal
lr_decay_iters = 300
min_lr = 3e-5
beta2 = 0.95

warmup_iters = 30 # ~10%
