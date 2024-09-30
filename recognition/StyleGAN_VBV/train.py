import os
import torch
from torch import optim
from tqdm import tqdm
from dataset import get_loader
from modules import Generator, Discriminator

# Hyperparameters
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
CHANNELS_IMG = 3
LR = 1e-3
LAMBDA_GP = 10
PROGRESSIVE_EPOCHS = [30] * 6  # Adjust based on your image sizes

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'