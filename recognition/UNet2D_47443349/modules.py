"""
modules.py

Author: Alex Pitman
Student ID: 47443349
COMP3710 - HipMRI UNet2D Segmentation Project
Semester 2, 2024

Contains UNet2D network architecture.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, initial_features=64, n_layers=5):
        super(UNet2D, self).__init__()

        self.CHANNEL_DIM = 1
        features = initial_features
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) # For encoder downsample
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.encoder.append(Block(in_channels, features))
            in_channels = features # Output feature size for this layer is input for next layer
            features *= 2 # Double feature size each layer
        self.bottleneck = Block(in_channels, features)

        # Decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(n_layers - 1):
            upsample = nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2)
            block = Block(features, features // 2)
            self.decoder.append(upsample)
            self.decoder.append(block)
            features = features // 2 # Half feature size each layer
        
        # Output segmentation
        final_features = initial_features
        self.segmentation = nn.Conv2d(final_features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for block in self.encoder:
            x = block(x)
            skip_connections.append(x) # To concatenate with decoder
            x = self.max_pool(x) # Downsample
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x) # Upsample
            skip_connection = skip_connections[idx//2] # Output from corresponding encoder layer
            if x.shape != skip_connection.shape:
                # Height and width dimensions need to be the same
                x = TF.resize(x, size=skip_connection.shape[2:])
            concatenated = torch.cat((skip_connection, x), dim=self.CHANNEL_DIM) # Concatenate skip connection
            x = self.decoder[idx+1](concatenated) # Block

        return self.segmentation(x)


def test():
    x = torch.randn(1, 1, 256, 144)
    model = UNet2D(in_channels=1, out_channels=6, initial_features=64, n_layers=5)
    predictions = model(x)
    print(x.shape)
    print(predictions.shape)

if __name__ == "__main__":
    test()