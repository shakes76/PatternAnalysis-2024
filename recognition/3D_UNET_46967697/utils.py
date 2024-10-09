# FILE PATHS
DATASET_PATH = '../../../data_copy/HipMRI_Study_open'
KERAS_SLICES_PATH = DATASET_PATH + '/keras_slices_data'
SEMANTIC_LABELS_PATH = DATASET_PATH + '/semantic_labels_only'
SEMANTIC_MRS_PATH = DATASET_PATH + '/semantic_MRs'

# TRAINING PARAMETERS
RANDOM_SEED = 1
TRAIN_TEST_SPLIT = 0.8
EPOCHS = 2
BATCH_SIZE = 2
IN_DIM = 1
NUM_CLASSES = 6
NUM_FILTERS = 8

# HELPER FUNCTIONS
import os

def uncompress_nii_gz(directory):
    for subdir in os.listdir(directory):
        if not os.path.isdir(f'{directory}/{subdir}'):
            continue
        for file_name in os.listdir(f'{directory}/{subdir}'):
            if file_name.endswith('.nii.gz'):
                os.system(f'gunzip {directory}/{subdir}/{file_name}')