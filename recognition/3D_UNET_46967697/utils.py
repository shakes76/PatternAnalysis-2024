"""
Contains constants including file paths and parameters for the 3D U-Net model
Also contains helper functions for uncompressing nii.gz files

@author Damian Bellew
"""

# FILE PATHS
DATASET_PATH = '../../../data_copy/HipMRI_Study_open'
KERAS_SLICES_PATH = DATASET_PATH + '/keras_slices_data'
SEMANTIC_LABELS_PATH = DATASET_PATH + '/semantic_labels_only'
SEMANTIC_MRS_PATH = DATASET_PATH + '/semantic_MRs'
MODEL_PATH = '3d_unet_model.pth'
SAVED_IMAGES_PATH = 'images'
PREDICTION_PATH = SAVED_IMAGES_PATH + '/predictions'
ORIGINAL_IMAGES_PATH = SAVED_IMAGES_PATH + '/original_images'
ORIGINAL_LABELS_PATH = SAVED_IMAGES_PATH + '/original_labels'
DICE_LOSS_GRAPH_PATH = SAVED_IMAGES_PATH + '/dice_loss_graph.png'

# PARAMETERS
RANDOM_SEED = 1
TRAIN_TEST_SPLIT = 0.8
EPOCHS = 100
BATCH_SIZE = 2
# model parameters
IN_DIM = 1
NUM_CLASSES = 6
NUM_FILTERS = 8
# optimizer parameters
LR = 1e-5
WEIGHT_DECAY = 1e-8
# scheduler parameters
STEP_SIZE = 10
GAMMA = 0.1
# dice loss parameters
SMOOTH = 1e-6

# HELPER FUNCTIONS
import os

def uncompress_nii_gz(directory):
    """
    Uncompresses all .nii.gz files in the given directory and its subdirectories.
    
    directory (str): The root directory containing subdirectories with .nii.gz files.
    """
    for subdir in os.listdir(directory):
        if not os.path.isdir(f'{directory}/{subdir}'):
            continue
        for file_name in os.listdir(f'{directory}/{subdir}'):
            if file_name.endswith('.nii.gz'):
                os.system(f'gunzip {directory}/{subdir}/{file_name}')