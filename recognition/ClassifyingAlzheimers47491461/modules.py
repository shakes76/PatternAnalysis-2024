import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import dataset

class PatchyEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, latent_size=768, batch_size=32):
        super(PatchyEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.batch_size = batch_size

        #self.num_patches = (img_size // patch_size) ** 2
        #self.patch_dim = in_channels * patch_size * patch_size  # Flattened size of each patch


        #self.proj = nn.Linear(self.patch_dim, embed_dim)
        # Downsample the images (project patches into embedded space)
        # More efficient than image worth more 16x16 paper??
        self.project = nn.Conv2d(3, self.latent_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        '''
        batch_size, _, height, width = x.shape

        x = x.reshape(batch_size, 3, height // self.patch_size, self.patch_size, width // self.patch_size,
                      self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).flatten(2, 3)
        '''
        x = self.project(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim, num_patches, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-np.log(10000.0) / latent_dim))
        pe = torch.zeros(1, num_patches, latent_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x