# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-image-mdct'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 1

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'image-mdct'
wandb_run_name = 'image-mdct'

dataset = 'image_mdct'
block_size = 1024*4
gradient_accumulation_steps = 1
batch_size = 64 # rows/step

block_type = "mamba"
n_layer = 10 # Mamba2 block count
n_head = 8 # SSM “heads” (not attention); with headdim=64 → d_ssm=512
n_embd = 512 # d_model
n_inner = 1024 # if you have an MLP head; otherwise ignore
n_conv = 4 # conv kernel (selective conv) per block
n_state = ([64]*8) + ([96]*2)
n_chunk = 32 # scan chunk/unroll (kernel perf knob)

dropout = 0.05 # or 0.0 if you use even light aug
learning_rate = 3e-4 # AdamW
max_iters = 12000 # ≈ one pass compute-optimal
lr_decay_iters = 300
min_lr = 3e-5
beta2 = 0.95

warmup_iters = 30 # ~10%
