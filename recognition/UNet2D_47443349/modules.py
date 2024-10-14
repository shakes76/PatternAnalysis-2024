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
        layers = [] # Each layer of the Encoder is a Block with a downsample
        for feature_size in feature_sizes:
            layers.append(Block(in_channels, feature_size))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # Downsample
            in_channels = feature_size # Current feature_size will be the in_channels for next layer
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, feature_sizes):
        super().__init__()
        layers = []
        # Reverse through the feature_sizes used for Encoder
        for i in range(len(feature_sizes) - 1, 0, -1):
            layers.append(nn.ConvTranspose2d(feature_sizes[i], feature_sizes[i-1], kernel_size=2, stride=2)) # Upsample
            layers.append(Block(feature_sizes[i], feature_sizes[i-1]))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)



    
	