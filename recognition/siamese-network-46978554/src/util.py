from pathlib import Path

# Dataset directory
# We use a downsized version of the ISIC 2020 dataset:
# https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data
DATA_DIR = Path(__file__).parent.parent / "data"

# Model and loss output directory
OUT_DIR = Path(__file__).parent.parent / "out"
