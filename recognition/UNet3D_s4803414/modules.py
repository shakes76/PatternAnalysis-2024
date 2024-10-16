import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Contracting path
        self.enc1 = self.contracting_block(in_channels, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)

        # Bottleneck
        self.bottleneck = self.contracting_block(512, 1024)

        # Expanding path
        self.dec4 = self.expanding_block(1024, 512)
        self.dec3 = self.expanding_block(512, 256)
        self.dec2 = self.expanding_block(256, 128)
        self.dec1 = self.expanding_block(128, 64)
# Final output layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)  # Downsampling
        )

    def expanding_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsampling
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoding path
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoding path
        dec4 = self.dec4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Skip connection
        dec3 = self.dec3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Skip connection
        dec2 = self.dec2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec1 = self.dec1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Skip connection

        return self.final_conv(dec1)


