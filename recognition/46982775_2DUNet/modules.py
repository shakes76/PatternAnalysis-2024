""" 
Classes to construct the 2D UNet architecture to perform segmentation.

2D UNet consists of double convolutions, maxpooling, upconvolutions,
then finally the 1x1 output convolution to produced segmented data.

Authors:
    Joseph Reid

Classes:
    DoubleConvolution: Double convolution block
    MaxPool: Maxpooling block to downsample
    UpConvolution: Upconvolution block to upsample
    OutConvolution: Output convolution block to produce segmented data
    UNet: 2D UNet model made up of the previous blocks

Dependencies:
    pytorch
    torchvision
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConvolution(nn.Module):
    """ Double convolution with batch norm and ReLU activation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # Second convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class MaxPool(nn.Module):
    """ Downsample with max pooling during contracting path."""

    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.max_pool(x)


class UpConvolution(nn.Module):
    """ Upsample with up-convolution during expanding path."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up_conv(x)


class OutConvolution(nn.Module):
    """ 1x1 Output convolution to map to the desired number of classes."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)


class UNet(nn.Module):
    """
    2D UNet architecture made up of various NN blocks.
    
    Follows the typical, original UNet architecture with 5 layers,
    where the input image is downsampled 4 times before reaching the
    bottleneck layer, then upsampled 4 times before forming the
    output segmentation map.

    Attributes:
        in_channels: Input channels, eg. 1 for grey-scale images
        out_channels: Number of classes to segment
        n_features: Number of features after first convolution

    Methods:
        forward: Forward pass that returns the output segmentation
    """

    def __init__(self, in_channels=1, out_channels=6, n_features=64):
        super().__init__()
        # Contracting path (encoder)
        self.pool = MaxPool() # Downsampling used in each layer
        self.enc_conv1 = DoubleConvolution(in_channels, n_features)
        self.enc_conv2 = DoubleConvolution(n_features, n_features * 2)
        self.enc_conv3 = DoubleConvolution(n_features * 2, n_features * 4)
        self.enc_conv4 = DoubleConvolution(n_features * 4, n_features * 8)
        self.bottleneck = DoubleConvolution(n_features * 8, n_features * 16)
        # Expanding path (decoder)
        self.up_conv1 = UpConvolution(n_features * 16, n_features * 8)
        self.dec_conv1 = DoubleConvolution(n_features * 16, n_features * 8)
        self.up_conv2 = UpConvolution(n_features * 8, n_features * 4)
        self.dec_conv2 = DoubleConvolution(n_features * 8, n_features * 4)
        self.up_conv3 = UpConvolution(n_features * 4, n_features * 2)
        self.dec_conv3 = DoubleConvolution(n_features * 4, n_features * 2)
        self.up_conv4 = UpConvolution(n_features * 2, n_features)
        self.dec_conv4 = DoubleConvolution(n_features * 2, n_features)
        # Output convolution
        self.out_conv = OutConvolution(n_features, out_channels)

    def forward(self, x):
        original_shape = x.shape # (Batch[B], 1, 256, 256)

        # Encoding path, layers from up to down:
        # Layer 1
        x = self.enc_conv1(x) # (B, 64, 252, 252)
        skip1 = x
        x = self.pool(x) # (B, 64, 126, 126)
        # Layer 2
        x = self.enc_conv2(x) # (B, 128, 122, 122)
        skip2 = x
        x = self.pool(x) # (B, 128, 61, 61)
        # Layer 3
        x = self.enc_conv3(x) # (B, 256, 57, 57)
        skip3 = x
        x = self.pool(x) # (B, 256, 28, 28)
        # Layer 4
        x = self.enc_conv4(x) # (B, 512, 24, 24)
        skip4 = x
        x = self.pool(x) # (B, 512, 12, 12)
        # Layer 5, bottleneck layer
        x = self.bottleneck(x) # (B, 1024, 8, 8)
        x = self.up_conv1(x) # (B, 512, 16, 16)

        # Decoding path, layers from down to up:
        # Layer 1
        skip4 = TF.center_crop(skip4, x.shape[2:]) # (24, 24) to (16, 16)
        x = torch.cat((skip4, x), dim=1) # (B, 1024, 16, 16)
        x = self.dec_conv1(x) # (B, 512, 12, 12)
        x = self.up_conv2(x) # (B, 256, 24, 24)
        # Layer 2
        skip3 = TF.center_crop(skip3, x.shape[2:]) # (57, 57) to (24, 24)
        x = torch.cat((skip3, x), dim=1) # (B, 512, 24, 24)
        x = self.dec_conv2(x) # (B, 256, 20, 20)
        x = self.up_conv3(x) # (B, 128, 40, 40)
        # Layer 3
        skip2 = TF.center_crop(skip2, x.shape[2:]) # (122, 122) to (40, 40)
        x = torch.cat((skip2, x), dim=1) # (B, 256, 40, 40)
        x = self.dec_conv3(x) # (B, 128, 36, 36)
        x = self.up_conv4(x) # (B, 64, 72, 72)
        # Layer 4, output layer
        skip1 = TF.center_crop(skip1, x.shape[2:]) # (252, 252) to (72, 72)
        x = torch.cat((skip1, x), dim=1) # (B, 128, 72, 72)
        x = self.dec_conv4(x) # (B, 64, 68, 68)
        # Output
        x = self.out_conv(x) # (B, 6, 68, 68)
        x = TF.resize(x, original_shape[2:], antialias=None, interpolation=0)
        return x # (B, 6, 256, 256)


# For testing purposes
if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    model = UNet()
    print(model(x).shape)