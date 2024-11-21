""" 
File: modules.py
Author: Øystein Kvandal
Description: Contains the 2D UNET model for the medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2D(nn.Module):
    """ 
    2D UNet model for the medical image segmentation.
    """
    def __init__(self):
        super(UNet2D, self).__init__()

        # Downstream
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv2d(1, 32)
        self.down_conv_2 = double_conv2d(32, 64)
        self.down_conv_3 = double_conv2d(64, 128)
        self.down_conv_4 = double_conv2d(128, 256)

        # Upstream
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_1 = double_conv2d(256, 128)
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_2 = double_conv2d(128, 64)
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=2,
                                            stride=2)
        self.up_conv_3 = double_conv2d(64, 32)
        self.out = nn.Conv2d(in_channels=32,
                            out_channels=6,
                            kernel_size=1)
    

    def forward(self, image):
        # Encoding
        x1 = self.down_conv_1(image)
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)

        ### Decoding
        x = self.up_trans_1(x7)
        x = self.up_conv_1(torch.cat([x, x5], 1))
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x, x3], 1))
        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x, x1], 1))

        x = self.out(x)
        return x


def double_conv2d(in_channels, out_channels):
    """
    Double convolutional layer with batch normalization and ReLU activation for the layers in the UNet model.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class DiceLoss(nn.Module):
    """ 
    Dice loss calculation for the medical image segmentation.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-14

    def forward(self, inputs, targets, return_dice=False, separate_classes=False):
        """ 
        Calculate the dice loss.

        Args:
            inputs (torch.Tensor): The input tensor of format (B, C, H, W) with C being the number of classes (6).
            targets (torch.Tensor): The target tensor of format (B, C, H, W) with values in the range [0, 5].
            return_dice (bool): Whether to return the dice coefficient rather than the loss.

        Returns:
            torch.Tensor: The dice loss (or coefficient).
            list: The dice coefficient for each class if separate_classes is True.
        """
        inputs = torch.softmax(inputs, dim=1)        
        targets = F.one_hot(targets, num_classes=6)
        targets = targets.squeeze(1)
        targets = targets.permute(0, 3, 1, 2).float()

        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(inputs.size(0), inputs.size(1), -1)

        if separate_classes:
            intersect = (inputs * targets).sum(-1)
            inputs_sum = inputs.sum(-1)
            targets_sum = targets.sum(-1)
            dice = (2 * intersect + self.smooth) / (inputs_sum + targets_sum + self.smooth)
            return dice if return_dice else 1 - dice
        
        intersect = torch.abs(inputs * targets).sum()
        dice = (2 * intersect + self.smooth) / (torch.abs(inputs).sum(-1).sum() + torch.abs(targets).sum(-1).sum() + self.smooth)
        
        if return_dice:
            return dice.mean()
        return 1 - dice.mean()
