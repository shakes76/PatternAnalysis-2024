import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DoubleConv(nn.Module):
    """
    Applies two consecutive convolutional layers with ReLU activation and optional batch normalisation.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, use_batchnorm: bool = True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ])
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)