# modules.py
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):  # Added dropout_prob
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_prob)  # Dropout layer added

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Apply dropout after activation
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class Improved3DUNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):  # Added dropout_prob
        super(Improved3DUNet, self).__init__()
        self.encoder1 = ResidualBlock(in_channels, 64, dropout_prob)  # Pass dropout_prob
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = ResidualBlock(64, 128, dropout_prob)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = ResidualBlock(128, 256, dropout_prob)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ResidualBlock(256, 128, dropout_prob)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ResidualBlock(128, 64, dropout_prob)

        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        dec2 = self.upconv2(enc3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)
