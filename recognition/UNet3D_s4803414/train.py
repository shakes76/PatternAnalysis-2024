import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRIDataset
from modules import UNet3D
from torchvision import transforms

IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_png_slices_train'
MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_png_slices_seg_train'
MODEL_SAVE_PATH = '/home/Student/s4803414/miniconda3/model/model.pth'

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4

# Create dataset and dataloader
dataset = MRIDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)