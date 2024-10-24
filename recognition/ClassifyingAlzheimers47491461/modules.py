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
import torch.fft

class PatchyEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, latent_size=768, batch_size=32):
        super(PatchyEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.batch_size = batch_size

        # Downsample the images (project patches into embedded space)
        # More efficient than image worth more 16x16 paper??
        self.project = nn.Conv2d(3, self.latent_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
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


class GlobalFilterLayer(nn.Module): # broken
    def __init__(self, embed_dim):
        super(GlobalFilterLayer, self).__init__()
        # Learnable global filters in the frequency domain for each embedding dimension
        self.global_filter = nn.Parameter(torch.randn(1, embed_dim, 1, 1, dtype=torch.cfloat))  # (1, embed_dim, 1, 1)

    def forward(self, x):
        # Input shape (batch_size, num_tokens, embed_dim)
        # Reshape input to (batch_size, embed_dim, num_tokens, 1)
        x = x.transpose(1, 2).unsqueeze(-1)

        freq_x = torch.fft.fft2(x, dim=(-3, -2))

        filtered_freq_x = freq_x * self.global_filter

        output = torch.fft.ifft2(filtered_freq_x, dim=(-3, -2))

        output = output.squeeze(-1).transpose(1, 2)
        # Dimensionality issue needs debugging

        return output.real



class GFNetBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super(GFNetBlock, self).__init__()
        self.global_filter = GlobalFilterLayer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Global filtering in the frequency domain
        filtered_x = self.global_filter(self.norm1(x))
        x = x + self.dropout(filtered_x)

        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        return x


