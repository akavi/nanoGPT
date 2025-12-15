tmux
git clone --depth 1 --branch  trainable-scaling https://github.com/akavi/nanoGPT.git
cd nanoGPT/
sudo snap install astral-uv --classic
uv python install 3.13
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate
python -V   # should say 3.13.x
uv sync
uv run python3 config/face_fft_config.py

uv run python3 sample.py --out_dir=out-image-mdct-early --max_new_tokens=4096
uv run python3 sample.py --out_dir=out-image-mdct-even --max_new_tokens=4096
uv run python3 sample.py --out_dir=out-image-mdct-late --max_new_tokens=4096
uv run python3 sample.py --out_dir=out-image-mdct-superlate --max_new_tokens=4096

export LOGIN=root@86.38.238.45
export DIR=out-face-mdct-zigzag
scp -r -P 22 $LOGIN:~/nanoGPT/$DIR .
scp -r -P 22 out-face-mdct-zigzag $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-even $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-late $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-superlate $LOGIN:~/nanoGPT

scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-even/ .
scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-late/ .
scp -r -P 22 root@86.38.238.193:~/nanoGPT/out-image-mdct-hyperlate/ .
