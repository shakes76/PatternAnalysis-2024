import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    """
    U-Net model for medical image segmentation consisting of an encoder (contracting path),
    a bottleneck, and a decoder (expansive path).

    Attributes:
        in_channels (int): Number of input channels (1 for grayscale).
        out_channels (int): Number of output channels (1 for binary segmentation).
        retainDim (bool): If True, resizes output to match outSize dimensions.
        outSize (tuple): The desired output size (height, width) for the final output.
    """

    def __init__(self, in_channels=1, out_channels=1, retainDim=True, outSize=(256, 128), dropout_rate=0.3):
        """
        Initializes U-Net model by defining the encoder, bottleneck, and decoder layers.

        Args:
            in_channels (int): Number of input channels for the model.
            out_channels (int): Number of output channels for the model.
            retainDim (bool): If True, resizes output to match outSize dimensions.
            outSize (tuple): Target dimensions (height, width) for the final output.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(UNet, self).__init__()
        self.retainDim = retainDim
        self.outSize = outSize
        self.dropout_rate = dropout_rate

        # Encoder (contracting path)
        self.conv1 = self.double_conv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = self.double_conv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = self.double_conv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Bottleneck
        self.bottleneck = self.double_conv(64, 128)
        self.bottleneck_dropout = nn.Dropout(dropout_rate)
        
        # Decoder (expansive path)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = self.double_conv(128, 64)
        self.dropout_up3 = nn.Dropout(dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = self.double_conv(64, 32)
        self.dropout_up2 = nn.Dropout(dropout_rate)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up1 = self.double_conv(32, 16)
        self.dropout_up1 = nn.Dropout(dropout_rate)

        # Output layer
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Defines forward pass of U-Net model using encoder, bottleneck, and decoder.

        Args:
            x : Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor (dimensions are resized based on retainDim).
        """
        # Contracting path
        c1 = self.conv1(x)
        p1 = self.dropout1(self.pool1(c1))

        c2 = self.conv2(p1)
        p2 = self.dropout2(self.pool2(c2))

        c3 = self.conv3(p2)
        p3 = self.dropout3(self.pool3(c3))

        # Bottleneck
        bn = self.bottleneck_dropout(self.bottleneck(p3))

        # Expansive path
        up3 = self.upconv3(bn)
        up3 = torch.cat([up3, c3], dim=1)  # Skip connection
        up3 = self.dropout_up3(self.conv_up3(up3))

        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, c2], dim=1)  # Skip connection
        up2 = self.dropout_up2(self.conv_up2(up2))

        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, c1], dim=1)  # Skip connection
        up1 = self.dropout_up1(self.conv_up1(up1))

        out = self.out_conv(up1)

        if self.retainDim:
            out = F.interpolate(out, size=self.outSize, mode='bilinear', align_corners=False)
        return out

    def double_conv(self, in_channels, out_channels):
        """
        Creates a double convolution block with batch normalization and ReLU activation.

        Args:
            in_channels (int): Number of input channels for double convolution.
            out_channels (int): Number of output channels for double convolution.

        Returns:
            nn.Sequential: A sequential of two convolution layers with batch normalization and ReLU activation.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
