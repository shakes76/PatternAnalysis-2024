# simple_unet.py
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(Conv2d => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional: Helps with training
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Optional
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = DoubleConv(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(128, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)          # [B, 64, H, W]
        x2 = self.down1(x1)       # [B, 64, H/2, W/2]
        x3 = self.conv1(x2)       # [B, 128, H/2, W/2]
        x4 = self.down2(x3)       # [B, 128, H/4, W/4]
        x5 = self.conv2(x4)       # [B, 256, H/4, W/4]
        x = self.up1(x5)           # [B, 128, H/2, W/2]
        x = torch.cat([x, x3], dim=1)  # [B, 256, H/2, W/2]
        x = self.conv3(x)          # [B, 128, H/2, W/2]
        x = self.up2(x)            # [B, 64, H, W]
        x = torch.cat([x, x1], dim=1)  # [B, 128, H, W]
        x = self.conv4(x)          # [B, 64, H, W]
        logits = self.outc(x)      # [B, n_classes, H, W]
        return logits
