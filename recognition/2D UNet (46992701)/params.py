"""
This file contains definitions for various parameters for the model such as file paths  
and hyperparameter values 
"""

# Dataset paths
TRAIN_IMG_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
TRAIN_MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train'

TEST_IMG_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
TEST_MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test'

VAL_IMG_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'
VAL_MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate'

IMAGE_SIZE = (128, 128)

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 40


CHECKPOINT_DIR = 'checkpoints/checkpoint.pth.tar'
