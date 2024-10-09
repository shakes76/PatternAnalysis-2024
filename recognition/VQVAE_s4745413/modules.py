import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Reduces dimensionality of input MRI image
    """
    def __init__(self, in_channels, hidden_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1) # downamplng
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, latent_dim, kernel_size=3, stride=1, padding=1)

        # Need to add Residual Stack here

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim: input dimension
    - h_dim: hidden layer dimension
    - res_h_dim: hidden dimension of residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True)
                nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)

        def forward(self, x):
        x = x + self.res_block(x)
        return x

