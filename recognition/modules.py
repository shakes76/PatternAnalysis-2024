# MODULES.PY

# IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    A module consisting of two convolutional layers each followed by BatchNorm and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))

        layers.extend([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        if dropout_p > 0:
            layers.append(nn.Dropout2d(p=dropout_p))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class SimpleUNet(nn.Module):
    """
    A simplified UNet model for semantic segmentation to reduce overfitting.
    """
    def __init__(self, n_channels, n_classes, dropout_p=0.0):
        super(SimpleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Reduced number of filters and depth
        self.inc = DoubleConv(n_channels, 32, dropout_p)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(32, 64, dropout_p)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128, dropout_p)
        )

        # Expanding path
        self.up1_convtrans = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1_conv = DoubleConv(128, 64, dropout_p)

        self.up2_convtrans = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up2_conv = DoubleConv(64, 32, dropout_p)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)      # Initial conv
        x2 = self.down1(x1)   # Downsample 1
        x3 = self.down2(x2)   # Downsample 2

        # Expanding path
        x = self.up1_convtrans(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1_conv(x)

        x = self.up2_convtrans(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2_conv(x)

        logits = self.outc(x)
        return logits
