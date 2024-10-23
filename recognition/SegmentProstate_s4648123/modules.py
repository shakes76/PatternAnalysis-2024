"""
Normal Difficulty
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Ozgun Cicek et al.
"""
from torch import nn
import torch


class Conv3DBlock(nn.Module):
    """
    A 3D convolutional block consisting of two 3x3x3 convolutions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bottleneck (bool): If True, disables pooling to act as a bottleneck block.

    Forward:
        input (torch.Tensor): Input tensor to be convolved.

    Returns:
        (torch.Tensor, torch.Tensor):
            - Output after convolution (and pooling).
            - Residual output for potential skip connections.
    """

    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck

        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))

        out = self.pooling(res) if not self.bottleneck else res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    A 3D upsampling block with optional final output layer and skip connections.

    Args:
        in_channels (int): Number of input channels.
        res_channels (int, optional): Channels from skip connections to concatenate. Defaults to 0.
        last_layer (bool, optional): Whether this block is the final output layer. Defaults to False.
        num_classes (int, optional): Number of output channels for classification. Required if last_layer=True.

    Forward:
        input (torch.Tensor): Input tensor to upsample.
        residual (torch.Tensor, optional): Residual tensor to concatenate with input. Defaults to None.

    Returns:
        torch.Tensor: Final output tensor after upsampling and convolution.
    """
    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super().__init__()
        assert not last_layer or (last_layer and num_classes is not None), \
            "num_classes must be specified if last_layer=True."

        self.upconv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels + res_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1)

        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, input, residual=None):
        out = self.upconv(input)

        if residual is not None:
            out = torch.cat((out, residual), dim=1)

        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))

        if self.last_layer:
            out = self.conv3(out)

        return out


class UNet3D(nn.Module):
    """
    A 3D U-Net model with encoding and decoding paths.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output channels or segmentation masks.
        level_channels (list of int): Number of channels at each level in the encoder.
        bottleneck_channel (int): Number of channels in the bottleneck layer.

    Forward:
        input (torch.Tensor): Input tensor to the U-Net model.

    Returns:
        torch.Tensor: Output tensor after passing through the U-Net.
    """
    def __init__(self, in_channels, num_classes, level_channels=(64, 128, 256), bottleneck_channel=512) -> None:
        super().__init__()

        self.a_block1 = Conv3DBlock(in_channels, level_channels[0])
        self.a_block2 = Conv3DBlock(level_channels[0], level_channels[1])
        self.a_block3 = Conv3DBlock(level_channels[1], level_channels[2])
        self.bottleNeck = Conv3DBlock(level_channels[2], bottleneck_channel, bottleneck=True)

        self.s_block3 = UpConv3DBlock(bottleneck_channel, level_channels[2])
        self.s_block2 = UpConv3DBlock(level_channels[2], level_channels[1])
        self.s_block1 = UpConv3DBlock(level_channels[1], level_channels[0], num_classes=num_classes, last_layer=True)

    def forward(self, input):
        # Encoder (Analysis path)
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Decoder (Synthesis path)
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)

        return out
