import torch
import numpy as np
import random


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 144

TRAIN_IMG_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
TRAIN_MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
VAL_IMG_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
VAL_MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"
TEST_IMG_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
TEST_MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

SEED = 47443349
def set_seed(seed=SEED):
    """
    For reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)