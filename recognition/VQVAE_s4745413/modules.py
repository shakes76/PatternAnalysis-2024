import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    """
    Reduces dimensionality of input MRI image
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1), # downsampling
                nn.ReLU(),
                nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
                )

    def forward(self, x):
        return self.conv_stack(x)

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
                nn.ReLU(True),
                nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
                )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers:
    - in_dim: input dimension - h_dim: hidden layer dimension
    - res_h_dim: hidden dimension of residual block
    - n_res_layers: number of layers to stack
    """
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
            x = F.relu(x)
        return x


