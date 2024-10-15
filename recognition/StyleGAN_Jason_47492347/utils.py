import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from math import log2, sqrt

# set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# relative paths to dataset
test_path = '../ADNI_AD_NC_2D/AD_NC/test/NC'
train_path = '../ADNI_AD_NC_2D/AD_NC/train/NC'

# hyperparameters
IMG_SIZE = 128
START_SIZE = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
LOG_RES = int(log2(IMG_SIZE))
DIM_Z = 256
DIM_W = 256
CHANNELS = 1
