# relative path to dataset
DATASET = '../ADNI_AD_NC_2D/AD_NC/test'

# hyperparameters
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
LEARNING_RATE           = 1e-3
BATCH_SIZES             = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG            = 3
Z_DIM                   = 256
W_DIM                   = 256
IN_CHANNELS             = 256
LAMBDA_GP               = 10
PROGRESSIVE_EPOCHS      = [30] * len(BATCH_SIZES)
IMG_SIZE                = 128