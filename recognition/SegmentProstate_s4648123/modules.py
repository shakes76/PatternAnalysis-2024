import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Two consecutive 3x3x3 convolutions with BatchNorm and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the double convolution block."""
        return self.conv(x)


class Down(nn.Module):
    """Downsampling block with max pooling followed by DoubleConv.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after the convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Forward pass through the downsampling block."""
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsampling block with transposed convolution and DoubleConv.

    Args:
        in_channels (int): Number of input channels for the transposed convolution.
        out_channels (int): Number of output channels after the convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass through the upsampling block.

        Args:
            x1 (Tensor): Input feature map from the previous layer.
            x2 (Tensor): Corresponding feature map from the analysis path (for skip connections).

        Returns:
            Tensor: Concatenated feature map after upsampling and convolution.
        """
        # Perform upsampling
        x1 = self.up(x1)

        # Handle size mismatches between x1 and x2 (if any)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        # Pad the smaller tensor to match the size of the larger one
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final layer with 1x1x1 convolution to map to output channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (number of classes).
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the output convolution."""
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net architecture for volumetric data segmentation.

    Args:
        n_channels (int): Number of input channels (1).
        n_classes (int): Number of output classes (label_files).
    """
    def __init__(self, n_channels, n_classes):
        super(UNet3D, self).__init__()

        # Define the architecture with increasing and decreasing channels
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        """Forward pass through the 3D U-Net model.

        Args:
            x (Tensor): Input tensor with shape (batch_size, channels, height, width, depth).

        Returns:
            Tensor: Output tensor with shape (batch_size, n_classes, height, width, depth).
        """
        # Analysis path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Synthesis path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Final output layer
        logits = self.outc(x)
        return logits
