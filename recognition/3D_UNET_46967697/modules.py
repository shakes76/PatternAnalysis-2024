"""
This file contains the 3D U-Net architecture.
The 3D U-Net architecture is implemented using the DoubleConv, Down, and Up modules.

@author Damian Bellew
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class DoubleConv(nn.Module):
    """
    A block that performs two consecutive 3D convolution operations with 
    batch normalization and ReLU activation.

    in_dim (int): Number of input channels.
    out_dim (int): Number of output channels.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv module.

        x: Input tensor
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block that applies a 3D max pooling operation followed by DoubleConv.

    in_dim (int): Number of input channels.
    out_dim (int): Number of output channels.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_dim, out_dim)
        )

    def forward(self, x):
        """
        Forward pass of the Down module.

        x: Input tensor
        """
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """
    Upsampling block that applies a transposed 3D convolution followed by DoubleConv.

    in_dim (int): Number of input channels.
    out_dim (int): Number of output channels.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_dim, out_dim)

    def forward(self, x1, x2):
        """
        Forward pass of the Up module.

        x1: Input tensor
        x2: Corresponding feature map from the contracting path for concatenation
        """
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Unet3D(nn.Module):
    """
    The full 3D U-Net architecture for volumetric segmentation.

    in_dim (int): Number of input channels.
    num_classes (int): Number of output channels (number of classes).
    num_filters (int): Number of filters for the first convolution layer (doubles after each downsampling).
    """
    def __init__(self, in_dim, num_classes, num_filters):
        super(Unet3D, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_filters = num_filters

        # Down sampling
        self.input = DoubleConv(in_dim, num_filters)

        self.down_1 = Down(num_filters, num_filters*2)
        self.down_2 = Down(num_filters*2, num_filters*4)
        self.down_3 = Down(num_filters*4, num_filters*8)
        self.down_4 = Down(num_filters*8, num_filters*16)

        # Up sampling
        self.up_1 = Up(num_filters*16, num_filters*8)
        self.up_2 = Up(num_filters*8, num_filters*4)
        self.up_3 = Up(num_filters*4, num_filters*2)
        self.up_4 = Up(num_filters*2, num_filters)

        self.output = nn.Conv3d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the 3D U-Net architecture.

        x: Input tensor
        """
        # Down sampling
        x1 = self.input(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        # Up sampling
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)

        # Output layer
        return self.output(x)
    

class DiceLoss(nn.Module):
    """
    Dice loss function. Returns the Dice loss for each class.

    smooth (float): Smoothing factor to avoid division by zero.
    """
    def __init__(self, smooth=SMOOTH):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass for Dice loss calculation.

        inputs: Predicted segmentation maps
        targets: Actual segmentation maps.
        """
        inputs = torch.softmax(inputs, dim=1)

        dice_losses = []

        for c in range(NUM_CLASSES):
            input_flat = inputs[:, c].contiguous().view(-1)
            target_flat = targets[:, c].contiguous().view(-1)

            intersection = (input_flat * target_flat).sum()
            dice_coefficient = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)

            dice_losses.append(1 - dice_coefficient)

        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()

        dice_coefficient = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_coefficient, dice_losses
