""" Modules to implement U-Net on the Prostate Dataset.

"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

class DoubleConvolution(nn.Module):
    """ Double Convolution with Batch Norm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), # ! Padding due to too small a size in bottleneck layer
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
class MaxPool(nn.Module):
    """ Downsample with max pooling during contracting path."""
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.max_pool(x)
    
class UpConvolution(nn.Module):
    """ Upsample with up-convolution during expanding path."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up_conv(x)

class OutConvolution(nn.Module):
    """ 1x1 Output convolution to map to the desired number of classes."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    # Forward propagation
    def forward(self, x):
        return self.out_conv(x)

class UNet(nn.Module):
    """ 2D UNet architecture."""
    def __init__(self, in_channels=1, out_channels=6, n_features=64):
        super().__init__()
        # For contracting path
        self.pool = MaxPool() # Downsampling
        self.conv1d = DoubleConvolution(in_channels, n_features)
        self.conv2d = DoubleConvolution(n_features, n_features * 2)
        self.conv3d = DoubleConvolution(n_features * 2, n_features * 4)
        self.conv4d = DoubleConvolution(n_features * 4, n_features * 8)
        self.bottleneck = DoubleConvolution(n_features * 8, n_features * 16) # Bottleneck

        # For expanding path
        self.upconv1 = UpConvolution(n_features * 16, n_features * 8)
        self.conv1u = DoubleConvolution(n_features * 16, n_features * 8)
        self.upconv2 = UpConvolution(n_features * 8, n_features * 4)
        self.conv2u = DoubleConvolution(n_features * 8, n_features * 4)
        self.upconv3 = UpConvolution(n_features * 4, n_features * 2)
        self.conv3u = DoubleConvolution(n_features * 4, n_features * 2)
        self.upconv4 = UpConvolution(n_features * 2, n_features)
        self.conv4u = DoubleConvolution(n_features * 2, n_features)
        self.outconv = OutConvolution(n_features, out_channels)

    def forward(self, x):
        # Input is size (1, 256, 128)

        # Layer 1: Encoder
        x = self.conv1d(x) # Size (64, 256, 128)
        skip1 = x
        x = self.pool(x) # Size (64, 128, 64)

        # Layer 2: Encoder
        x = self.conv2d(x) # Size (128, 128, 64)
        skip2 = x
        x = self.pool(x) # Size (128, 64, 32)

        # Layer 3: Encoder
        x = self.conv3d(x) # Size (256, 64, 32)
        skip3 = x
        x = self.pool(x) # Size (256, 32, 16)

        # Layer 4: Encoder
        x = self.conv4d(x) # Size (512, 32, 16)
        skip4 = x
        x = self.pool(x) # Size (512, 16, 8)

        # Bottleneck layer
        x = self.bottleneck(x) # Size (1024, 16, 8)
        x = self.upconv1(x) # Size (512, 32, 16)

        # Layer 1: Decoder
        skip4 = TF.resize(skip4, size=x.shape[2:], antialias=None)
        x = torch.cat((skip4, x), dim=1)
        x = self.conv1u(x)
        x = self.upconv2(x)
        
        # Layer 2: Decoder
        skip3 = TF.resize(skip3, size=x.shape[2:], antialias=None)
        x = torch.cat((skip3, x), dim=1)
        x = self.conv2u(x)
        x = self.upconv3(x)

        # Layer 3: Decoder
        skip2 = TF.resize(skip2, size=x.shape[2:], antialias=None)
        x = torch.cat((skip2, x), dim=1)
        x = self.conv3u(x)
        x = self.upconv4(x)

        # Layer 4: Decoder
        skip1 = TF.resize(skip1, size=x.shape[2:], antialias=None)
        x = torch.cat((skip1, x), dim=1)
        x = self.conv4u(x)
        x = self.outconv(x) # Size (6, 256, 128)
        return x

# For testing purposes
if __name__ == "__main__":
    x = torch.randn(1, 1, 256, 128)
    model = UNet()
    model(x)