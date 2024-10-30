import os
import torch
from torch.utils.data import DataLoader
from dataset import ProstateMRIDataset
from modules import UNet
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'data'  # Root directory containing all data folders
TRAIN_IMAGE_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'segtrain')
VALID_IMAGE_DIR = os.path.join(DATA_DIR, 'valid')
VALID_MASK_DIR = os.path.join(DATA_DIR, 'segvalid')
TEST_IMAGE_DIR = os.path.join(DATA_DIR, 'test')
TEST_MASK_DIR = os.path.join(DATA_DIR, 'segtest')

BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

