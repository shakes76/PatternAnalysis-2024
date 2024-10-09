# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms # type: ignore
import matplotlib.pyplot as plt
from modules import VQVAE # type: ignore # Import your VQVAE model
from dataset import ProstateMRIDataset  # Import the custom dataset

# Hyperparameters
image_size = 256  # Image size for resizing
batch_size = 32  # Adjust this based on available memory
num_epochs = 35  # Number of training epochs
learning_rate = 0.0001  # Learning rate for optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
