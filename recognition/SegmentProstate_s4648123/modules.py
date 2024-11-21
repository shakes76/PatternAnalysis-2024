"""
    File name: modules.py
    Author: Elliana Rogaczewski-Novak
    Date last modified: 30/10/2024

    Description: This file implements components for a 3D U-Net architecture based on the work of
    Çiçek et al. (2016): "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
    (https://arxiv.org/abs/1606.06650).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(in_channels, out_channels) -> nn.Sequential:
    """
    Creates a convolutional block consisting of two convolutional layers followed by batch normalization
    and ReLU activation.
    """
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet3D(nn.Module):
    """
    A 3D U-Net architecture for volumetric data segmentation.

    Args:
        in_channel (int): Number of input channels. Default is 1.
        out_channel (int): Number of output channels (classes). Default is 6.
    """

    def __init__(self, in_channel= 1, out_channel= 6) -> None:
        super(UNet3D, self).__init__()

        # Analysis Path
        self.encoder1 = conv_block(in_channel, 32)
        self.encoder2 = conv_block(32, 64)
        self.encoder3 = conv_block(64, 128)
        self.encoder4 = conv_block(128, 256)

        # Synthesis Path
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = conv_block(64, 32)

        # Output layer
        self.output_conv = nn.Conv3d(32, out_channel, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W, D), where B is the batch size, C is the number of
                            channels, H is the height, W is the width, and D is the depth.

        Returns:
            torch.Tensor: Output tensor after passing through the U-Net.
        """
        # Analysis path with skip connections
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool3d(enc3, kernel_size=2, stride=2))

        # Synthesis path with up-convolutions and skip connections
        dec3 = self.upconv3(enc4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenate with encoder feature map
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # Final output layer
        output = self.output_conv(dec1)
        return output
