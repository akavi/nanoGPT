tmux
git clone --depth 1 --branch master https://github.com/akavi/nanoGPT.git
cd nanoGPT/
sudo snap install astral-uv --classic
uv python install 3.13
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate
python -V   # should say 3.13.x
uv sync

uv run python3 data/image_mdct/prepare.py  
uv run python3 train.py config/train_image_mdct.py --out_dir=out-image-mdct-top-heavy --batch_size=48 --max_iters=1000
uv run python3 sample.py --out_dir=out-image-mdct-early --max_new_tokens=4096
uv run python3 sample.py --out_dir=out-image-mdct-even --max_new_tokens=4096
uv run python3 sample.py --out_dir=out-image-mdct-late --max_new_tokens=4096
uv run python3 sample.py --out_dir=out-image-mdct-superlate --max_new_tokens=4096

export LOGIN=root@86.38.238.123
scp -r -P 22 out-image-mdct-early $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-even $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-late $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-superlate $LOGIN:~/nanoGPT

scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-even-large/ .
scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-even/ .
scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-late/ .
scp -r -P 22 root@86.38.238.193:~/nanoGPT/out-image-mdct-hyperlate/ .
