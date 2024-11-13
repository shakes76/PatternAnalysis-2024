"""
This file contains definitions for various parameters for the model such as file paths  
and hyperparameter values 
"""
import torch
import numpy as np

# Dataset paths
TRAIN_IMG_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
TRAIN_MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train'

TEST_IMG_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
TEST_MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test'

VAL_IMG_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'
VAL_MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate'

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 3
NUM_EPOCHS = 40


CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'

def dice_score(loader, device, model):
    with torch.no_grad():
        dice_score = 0
        for (x, y) in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float() # convert to binary mask
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8) # add 1e-8 to avoid division by 0
    dice_score = dice_score / len(loader)
    dice_score = dice_score.item()
    dice_score = np.round(dice_score, 4)
    return dice_score