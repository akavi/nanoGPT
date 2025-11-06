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
uv run python4 train.py config/train_image_mdct.py --init_from=resume --out_dir=out-image-mdct3 --batch_size=32 --learning_rate=2e-4
uv run python3 sample.py --out_dir=out-image-mdct --max_new_tokens=4096

scp -r -p 22 out-image-mdct3 root@86.38.238.183:~/nanoGPT

scp -r ubuntu@69.19.136.157:~/nanoGPT/out-image-mdct/ .
