tmux
git clone --depth 1 --branch master https://github.com/akavi/nanoGPT.git
cd nanoGPT/
sudo snap install astral-uv --classic
uv python install 3.13
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate
uv sync
uv run python3 config/face_ard_linear_raster_config.py --n_step=2 --latent_loss_scale=0.0 --n_embd=384

git fetch && git reset origin/master --hard


export LOGIN=root@31.22.104.32
export DIR=out-face-linear-raster
scp -r -P 22 $LOGIN:~/nanoGPT/$DIR .
scp -r -P 22 out-face-mdct-zigzag $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-even $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-late $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-superlate $LOGIN:~/nanoGPT

scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-even/ .
scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-late/ .
scp -r -P 22 root@86.38.238.193:~/nanoGPT/out-image-mdct-hyperlate/ .
