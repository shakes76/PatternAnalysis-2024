
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """两次卷积操作：Conv3D -> ReLU -> Conv3D -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ImprovedUNet3D(nn.Module):
    """改进的3D UNet模型"""
    def __init__(self, in_channels, out_channels):
        super(ImprovedUNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool3d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        self.out_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码路径
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        x4 = self.pool2(x3)
        x5 = self.down3(x4)
        x6 = self.pool3(x5)
        x7 = self.down4(x6)
        x8 = self.pool4(x7)

        # Bottle neck
        x9 = self.bottleneck(x8)

        # 解码路径
        x10 = self.up1(x9)
        x10 = torch.cat([x7, x10], dim=1)
        x11 = self.conv1(x10)
        x12 = self.up2(x11)
        x12 = torch.cat([x5, x12], dim=1)
        x13 = self.conv2(x12)
        x14 = self.up3(x13)
        x14 = torch.cat([x3, x14], dim=1)
        x15 = self.conv3(x14)
        x16 = self.up4(x15)
        x16 = torch.cat([x1, x16], dim=1)
        x17 = self.conv4(x16)

        output = self.out_conv(x17)
        return output

