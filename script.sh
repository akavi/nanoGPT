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
uv run python3 train.py config/train_image_raster.py --init_from=resume
uv run python3 sample.py --out_dir=out-image-mdct --max_new_tokens=4096
scp -r out-image-mdct ubuntu@69.19.136.157:~/nanoGPT
