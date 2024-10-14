import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
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
        return self.conv(x)
	

class Encoder(nn.Module):
    def __init__(self, in_channels, feature_sizes):
        """
        feature_sizes is a list with the number of channels/filters for each layer
        """
        super().__init__()
        self.operations = []
        # Each layer of the Encoder is a Block followed by a downsample
        for feature_size in feature_sizes:
            self.operations.append(Block(in_channels, feature_size))
            self.operations.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample
            in_channels = feature_size # Current feature_size will be the in_channels for next layer
        self.encoder = nn.Sequential(*self.operations)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, feature_sizes):
        super().__init__()
        self.operations = []
        # Reverse through the feature_sizes used for Encoder
        # Each layer of the Decoder is an upsample followed by a Block
        for i in range(len(feature_sizes) - 1, 0, -1):
            self.operations.append(nn.ConvTranspose2d(feature_sizes[i], feature_sizes[i-1], kernel_size=2, stride=2)) # Upsample
            self.operations.append(Block(feature_sizes[i], feature_sizes[i-1]))
        self.decoder = nn.Sequential(*self.operations)
    
    def forward(self, x):
        return self.decoder(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, feature_sizes=[64, 128, 256], channel_dim=1):
        super().__init__()
        self.channel_dim = channel_dim # Dimension for skip connection concatenation
        self.encoder = Encoder(in_channels, feature_sizes)
        self.bottleneck = nn.Sequential(
            Block(feature_sizes[-1], feature_sizes[-1] * 2),
            nn.ConvTranspose2d(feature_sizes[-1] * 2, feature_sizes[-1], kernel_size=2, stride=2)
        )
        self.decoder = Decoder(feature_sizes)
        self.segmentation = nn.Conv2d(feature_sizes[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc_outputs = []
        for i, operation in enumerate(self.encoder.operations):
            x = operation(x)
            # Save output of a block (i%2 == 0 in operations) for skip connections
            if i%2 == 0:
                enc_outputs.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, operation in enumerate(self.decoder.operations):
            x = operation(x)
            # Concatenate with corresponding encoder output (skip connection) after upsample (i%2 == 0 in operations)
            if i%2 == 0:
                enc_output = enc_outputs[-(i//2)-1]
                x = torch.cat((x, enc_output), dim=self.channel_dim)

        # Output raw logits
        return self.segmentation(x)

