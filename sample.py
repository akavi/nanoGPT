"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import math
import matplotlib.pyplot as plt
import os
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import sys


# -----------------------------------------------------------------------------
# Ratchet, but if it's dumb and it works it's, well, still a little dumb, but
# only a little
def load_callable(file_path: str, func_name: str):
    p = Path(file_path).resolve()
    mod_name = f"_dyn_{p.stem}_{abs(hash(p)) & 0xffff:x}"  # unique-ish to avoid cache collisions

    spec = spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {p}")

    mod = module_from_spec(spec)
    # Put in sys.modules so relative imports inside the file (if any) can work
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

    fn = getattr(mod, func_name, None)
    if not callable(fn):
        raise AttributeError(f"{func_name!r} not found or not callable in {p}")
    return fn
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**{**checkpoint['model_args'], "mode": "sample"})
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

data_path = os.path.join('data', checkpoint['config']['dataset'], 'prepare.py')
init_gen = load_callable(data_path, 'init_gen')
detokenize = load_callable(data_path, 'detokenize')
x = init_gen(device)
print(x)

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            state = model.initial_state(1)
            y = model.generate(x, max_new_tokens, state, temperature=temperature, top_k=top_k)
            print(f"{y.shape}, {max_new_tokens}")
            detokenize(y[0], os.path.join(out_dir, str(k)))
