"""
Configuration file: set hyperparameters and labels here.
"""


# relative path to dataset
DATASET = "../ADNI_AD_NC_2D_Combined/AD"

# relative path to model directory
SRC = "recognition/StyleGAN_Jason_47492347"

# hyperparameters
START_TRAIN_AT_IMG_SIZE = 8  # initial image resolution
LEARNING_RATE           = 1e-3
BATCH_SIZES             = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG            = 1  # for quicker computation and negligible data loss
Z_DIM                   = 256
W_DIM                   = 256
IN_CHANNELS             = 256
LAMBDA_GP               = 10  # gradient penalty for WGAN-GP loss
PROGRESSIVE_EPOCHS      = [12] * len(BATCH_SIZES)
IMG_SIZE                = 128  # final image resolution
MODEL_LABEL             = "12 Epoch Trial AD"
