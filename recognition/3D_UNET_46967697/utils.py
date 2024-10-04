# Constants

#DATASET_PATH = '/home/groups/comp3710/HipMRI_Study_open'
DATASET_PATH = '~/data_copy/HipMRI_Study_open'
KERAS_SLICES_PATH = DATASET_PATH + '/keras_slices_data'
SEMANTIC_LABELS_PATH = DATASET_PATH + '/semantic_labels_only'
SEMANTIC_MRS_PATH = DATASET_PATH + '/semantic_MRs'

RANDOM_SEED = 1
TRAIN_TEST_SPLIT = 0.8

import os

def uncompress_nii_gz(directory):
    for subdir in os.listdir(directory):
        for file_name in os.listdir(f'{directory}/{subdir}'):
            if file_name.endswith('.nii.gz'):
                os.system(f'gunzip {directory}/{subdir}/{file_name}')