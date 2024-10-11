import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tf
import torchvision
from torchvision.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train