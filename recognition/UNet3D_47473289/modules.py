# Source code of the components of the UNet3D Model.

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)

class ImprovedUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ImprovedUNet3D, self).__init__()
        print("UNET INIT")
        # Contracting path
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(512, 1024)

        # Expansive path
        self.upconv4 = UpConv(1024, 512)
        self.conv_up4 = ConvBlock(1024, 512)

        self.upconv3 = UpConv(512, 256)
        self.conv_up3 = ConvBlock(512, 256)

        self.upconv2 = UpConv(256, 128)
        self.conv_up2 = ConvBlock(256, 128)

        self.upconv1 = UpConv(128, 64)
        self.conv_up1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Contracting path
        x1 = self.conv1(x)
        #print("CONV1DONE")
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        #print("2")
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        #print("3")
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)
        #print("4")
        p4 = self.pool4(x4)

        # Bottleneck
        bn = self.bottleneck(p4)
        #print("BN")

        # Expansive path
        u4 = self.upconv4(bn)
        #print("upconv1")
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.conv_up4(u4)
        #print("conv up 1")

        u3 = self.upconv3(u4)
        #print("upconv2")
        u3 = torch.cat([u3, x3], dim=1)
        
        u3 = self.conv_up3(u3)
        #print("conv up 2")

        u2 = self.upconv2(u3)
        #print("upconv3")
        u2 = torch.cat([u2, x2], dim=1)
        
        u2 = self.conv_up2(u2)
        #print("conv up 3")

        u1 = self.upconv1(u2)
        #print("upconv4")
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv_up1(u1)
        #print("conv up 4")

        return self.final_conv(u1)
    
