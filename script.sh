tmux
git clone --depth 1 --branch master https://github.com/akavi/nanoGPT.git
cd nanoGPT/
sudo snap install astral-uv --classic
uv python install 3.13
rm -rf .venv
uv venv --python 3.13
source .venv/bin/activate
uv sync
uv run python3 config/face_ard_linear_raster_config.py --n_step=4 --latent_loss_scale=1.0 --n_embd=384

git fetch && git reset origin/master --hard


export LOGIN=ubuntu@216.81.248.153
export DIR=out-face-linear-raster
scp -r -P 22 $LOGIN:~/nanoGPT/$DIR .
scp -r -P 22 out-face-mdct-zigzag $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-even $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-late $LOGIN:~/nanoGPT
scp -r -P 22 out-image-mdct-superlate $LOGIN:~/nanoGPT

scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-even/ .
scp -r -P 22 $LOGIN:~/nanoGPT/out-image-mdct-late/ .
scp -r -P 22 root@86.38.238.193:~/nanoGPT/out-image-mdct-hyperlate/ .

step 0: model_mse=29023.9355, passthrough_mse=91.3328, ratio=317.7823
step 1: model_mse=48960.2422, passthrough_mse=171.9036, ratio=284.8122
