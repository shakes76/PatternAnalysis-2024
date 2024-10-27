# modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers each followed by BatchNorm and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    """
    UNet model for semantic segmentation.
    """
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels  # Number of input channels
        self.n_classes = n_classes    # Number of output classes

        # Contracting path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # Expanding path
        self.up1_convtrans = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(1024, 512)

        self.up2_convtrans = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(512, 256)

        self.up3_convtrans = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_conv = DoubleConv(256, 128)

        self.up4_convtrans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4_conv = DoubleConv(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)          # Initial conv
        x2 = self.down1(x1)       # Downsample 1
        x3 = self.down2(x2)       # Downsample 2
        x4 = self.down3(x3)       # Downsample 3
        x5 = self.down4(x4)       # Bottom layer

        # Expanding path
        x = self.up1_convtrans(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.up1_conv(x)

        x = self.up2_convtrans(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up2_conv(x)

        x = self.up3_convtrans(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up3_conv(x)

        x = self.up4_convtrans(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up4_conv(x)

        logits = self.outc(x)
        return logits
