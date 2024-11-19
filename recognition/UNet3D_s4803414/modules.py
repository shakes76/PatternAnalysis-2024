import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    A helper module that performs two sequential 3D convolutions, each followed by Batch Normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),  # First 3D convolution
            nn.BatchNorm3d(out_channels),  # Normalize the output of the convolution
            nn.ReLU(inplace=True),  # Apply ReLU activation
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),  # Second 3D convolution
            nn.BatchNorm3d(out_channels),  # Normalize the output of the second convolution
            nn.ReLU(inplace=True)  # Apply ReLU activation
        )

    def forward(self, x):
        return self.conv(x)  # Forward pass through the double convolution block


class ResNetBlock(nn.Module):
    """
    A residual block with two 3D convolutions. Includes a shortcut connection to add the input to the output,
    improving gradient flow and allowing deeper networks.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second convolution layer
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # Shortcut connection to match dimensions when needed (stride change or channel increase)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions differ, use a 1x1 convolution to adjust
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolution block
        out = self.conv2(out)
        out = self.bn2(out)

        # Add the shortcut connection (input) to the output
        out += self.shortcut(x)
        out = self.relu(out)  # Apply ReLU after adding the shortcut
        return out


class UNet3D(nn.Module):
    """
    A 3D U-Net architecture for segmentation, with residual blocks in the contracting path and skip connections
    between the contracting and expanding paths.
    """
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()

        # Contracting path: using ResNet blocks to extract features and downsample
        self.enc1 = self.contracting_block(in_channels, 64)   # First encoder block (input -> 64 channels)
        self.enc2 = self.contracting_block(64, 128)           # Second encoder block (64 -> 128 channels)
        self.enc3 = self.contracting_block(128, 256)          # Third encoder block (128 -> 256 channels)
        self.enc4 = self.contracting_block(256, 512)          # Fourth encoder block (256 -> 512 channels)

        # Bottleneck layer to connect the contracting and expanding paths
        self.bottleneck = ResNetBlock(512, 1024)  # Deepest block, increases channel size to 1024

        # Expanding path: upsample and combine with corresponding feature maps from contracting path
        self.dec4 = self.expanding_block(1024 + 512, 512)  # Expand and combine with enc4 features (1024+512 -> 512 channels)
        self.dec3 = self.expanding_block(512 + 256, 256)   # Expand and combine with enc3 features (512+256 -> 256 channels)
        self.dec2 = self.expanding_block(256 + 128, 128)   # Expand and combine with enc2 features (256+128 -> 128 channels)
        self.dec1 = self.expanding_block(128 + 64, 64)     # Expand and combine with enc1 features (128+64 -> 64 channels)

        # Final output layer to get the desired number of output channels (segmentation classes)
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)  # Output a segmentation map with out_channels

    def contracting_block(self, in_channels, out_channels):
        """
        Contracting path block: consists of a residual block followed by max pooling for downsampling.
        """
        return nn.Sequential(
            ResNetBlock(in_channels, out_channels),  # Residual block
            nn.MaxPool3d(kernel_size=2, stride=2)    # Max pooling to downsample by a factor of 2
        )

    def expanding_block(self, in_channels, out_channels):
        """
        Expanding path block: consists of a transposed convolution for upsampling, followed by a double convolution.
        """
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsample the input
            DoubleConv(out_channels, out_channels)  # Apply two 3D convolutions after upsampling
        )

    def forward(self, x):
        # Contracting path (encoder)
        enc1 = self.enc1(x)  # Pass input through first encoder block
        enc2 = self.enc2(enc1)  # Pass through second encoder block
        enc3 = self.enc3(enc2)  # Pass through third encoder block
        enc4 = self.enc4(enc3)  # Pass through fourth encoder block

        # Bottleneck (deepest layer in the network)
        bottleneck = self.bottleneck(enc4)

        # Expanding path (decoder)
        # Combine the bottleneck output with the corresponding feature map from the contracting path (skip connections)
        dec4_input = torch.cat((self.center_crop(bottleneck, enc4), enc4), dim=1)  # Concatenate bottleneck with enc4
        dec4 = self.dec4(dec4_input)  # Pass through the fourth decoder block

        dec3_input = torch.cat((self.center_crop(dec4, enc3), enc3), dim=1)  # Concatenate dec4 with enc3
        dec3 = self.dec3(dec3_input)  # Pass through the third decoder block

        dec2_input = torch.cat((self.center_crop(dec3, enc2), enc2), dim=1)  # Concatenate dec3 with enc2
        dec2 = self.dec2(dec2_input)  # Pass through the second decoder block

        dec1_input = torch.cat((self.center_crop(dec2, enc1), enc1), dim=1)  # Concatenate dec2 with enc1
        dec1 = self.dec1(dec1_input)  # Pass through the first decoder block

        return self.final_conv(dec1)  # Output the final segmentation map

    def center_crop(self, tensor, target_tensor):
        """
        Crops the input tensor to match the spatial size of the target tensor (used for skip connections).
        """
        _, _, d, h, w = target_tensor.size()  # Get spatial dimensions of the target tensor
        tensor_cropped = tensor[:, :, :d, :h, :w]  # Crop the tensor to match target dimensions
        return tensor_cropped  # Return the cropped tensor
