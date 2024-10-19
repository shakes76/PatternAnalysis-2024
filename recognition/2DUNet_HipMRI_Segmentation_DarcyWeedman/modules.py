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
    
class Down(nn.Module):
    """
    Downscaling with maxpool followed by double convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling with transposed convolution or bilinear upsampling followed by double convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, use_batchnorm: bool = True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            factor = 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            factor = 1
        self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Padding to handle odd input dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)