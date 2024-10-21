import torch

DATASET = "ADNI"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZES = [4, 8, 16, 32, 64, 128, 256]
BATCH_SIZES = {4: 612, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
LEARNING_SIZES = {4: 1e-3, 8: 1.2e-3, 16: 1.5e-3, 32: 1.8e-3, 64: 2e-3, 128: 2.5e-3, 256: 3e-3}
CHANNELS_IMG = 1  
Z_DIM = 256
W_DIM = 256
IN_CHANNELS = 256
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = {4: 50, 8: 50, 16: 40, 32: 30, 64: 20, 128: 15, 256: 10}  