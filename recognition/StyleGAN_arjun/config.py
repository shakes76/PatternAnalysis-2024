"""
Contains constants and hyperparameters for StyleGAN2 implementation.

This file defines:
    - Dataset paths and attributes
    - Model hyperparameters
    - Training configuration
"""

RANGPUR_PATH = "/home/groups/comp3710/" # Dataset path on Rangpur Cluster
PATH = '/Volumes/Acasis WD_Black/Documents/Deep Learning/Datasets/' # Dataset path on personal machine

# -- Dataset Attributes --
ADNI_TRAIN_PATH = "ADNI/AD_NC/train"
ADNI_TEST_PATH = "ADNI/AD_NC/test"
CIFAR_PATH = "cifar10/"

ADNI_IMG_SIZE = 64
CIFAR_IMG_SIZE = 32

# -- Hyper-parameters --
num_epochs = 50
lr = 1e-4 # Use 1e-3 for Adam
batch_size = 64
channels = 3
z_dim = 512
w_dim = 512
