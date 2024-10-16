import torch
import torch.nn as nn
import torch.nn.functional as F


# 3D Convolutional Block
class ConvBlock3D(nn.Module):
    """A block consisting of two 3D Convolutions with ReLU activations and Batch Normalization."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.batch_norm(self.conv1(x)))
        x = F.relu(self.batch_norm(self.conv2(x)))
        return x


# 3D Encoder (Downsampling)
class Encoder3D(nn.Module):
    """Encoder path that consists of 3D ConvBlocks followed by MaxPooling."""

    def __init__(self, in_channels, feature_channels):
        super(Encoder3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, feature_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        pooled = self.pool(x)
        return x, pooled


# 3D Decoder (Upsampling)
class Decoder3D(nn.Module):
    """Decoder path that consists of Upsampling followed by 3D ConvBlocks."""

    def __init__(self, in_channels, out_channels):
        super(Decoder3D, self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock3D(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_conv(x)
        # Concatenate along the channel dimension (skip connection)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv_block(x)
        return x


# Bottleneck
class Bottleneck3D(nn.Module):
    """Bottleneck part of the UNet architecture."""

    def __init__(self, in_channels, out_channels):
        super(Bottleneck3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)

    def forward(self, x):
        return self.conv_block(x)


# UNet3D Model
class UNet3D(nn.Module):
    """UNet3D architecture for 3D medical image segmentation."""

    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Encoder
        self.encoder1 = Encoder3D(in_channels, 64)
        self.encoder2 = Encoder3D(64, 128)
        self.encoder3 = Encoder3D(128, 256)
        self.encoder4 = Encoder3D(256, 512)

        # Bottleneck
        self.bottleneck = Bottleneck3D(512, 1024)

        # Decoder
        self.decoder4 = Decoder3D(1024, 512)
        self.decoder3 = Decoder3D(512, 256)
        self.decoder2 = Decoder3D(256, 128)
        self.decoder1 = Decoder3D(128, 64)

        # Output layer (Final 3D convolution)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1, pooled1 = self.encoder1(x)
        enc2, pooled2 = self.encoder2(pooled1)
        enc3, pooled3 = self.encoder3(pooled2)
        enc4, pooled4 = self.encoder4(pooled3)

        # Bottleneck
        bottleneck = self.bottleneck(pooled4)

        # Decoder path with skip connections
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Final convolution to map to the desired output classes
        return self.final_conv(dec1)
