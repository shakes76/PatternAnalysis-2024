import torch
import torch.nn as nn
# import numpy as np
# from torchvision import transforms

class NormConv3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation=nn.ReLU(),
    ):
        super(NormConv3D, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm = nn.BatchNorm3d(out_channels)

        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class NormConvTranspose3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation=nn.ReLU(),
    ):
        super(NormConvTranspose3D, self).__init__()

        # Convolutional layer
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.norm = nn.BatchNorm3d(out_channels)

        self.activation = activation

    def forward(self, x): 
        x = self.conv_transpose(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


# class UNet3D(nn.Module):
#     def __init__(self):
#         super(UNet3D, self).__init__()

#         # Encoder
#         self.enc11 = NormConv3D(1, 32, kernel_size=3, stride=1, padding=1)
#         self.enc12 = NormConv3D(32, 64, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

#         self.enc21 = NormConv3D(64, 64, kernel_size=3, stride=1, padding=1)
#         self.enc22 = NormConv3D(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

#         self.enc31 = NormConv3D(128, 128, kernel_size=3, stride=1, padding=1)
#         self.enc32 = NormConv3D(128, 256, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

#         # Bottleneck
#         self.botteneck1 = NormConv3D(256, 256, kernel_size=3, stride=1, padding=1)
#         self.botteneck2 = NormConv3D(256, 512, kernel_size=3, stride=1, padding=1)

#         # Decoder
#         self.up1 = NormConvTranspose3D(512, 512, kernel_size=2, stride=2)
#         self.dec11 = NormConv3D(768, 256, kernel_size=3, stride=1, padding=1)
#         self.dec12 = NormConv3D(256, 256, kernel_size=3, stride=1, padding=1)

#         self.up2 = NormConvTranspose3D(256, 256, kernel_size=2, stride=2)
#         self.dec21 = NormConv3D(384, 128, kernel_size=3, stride=1, padding=1)
#         self.dec22 = NormConv3D(128, 128, kernel_size=3, stride=1, padding=1)

#         self.up3 = NormConvTranspose3D(128, 128, kernel_size=2, stride=2)
#         self.dec31 = NormConv3D(192, 64, kernel_size=3, stride=1, padding=1)
#         self.dec32 = NormConv3D(64, 64, kernel_size=3, stride=1, padding=1)

#         self.final = nn.Conv3d(64, 6, kernel_size=1)  # TODO check if this is right

#     def forward(self, x):
#         # Encocder foward pass
#         e11 = self.enc11(x)
#         e12 = self.enc12(e11)
#         p1 = self.pool1(e12)

#         e21 = self.enc21(p1)
#         e22 = self.enc22(e21)
#         p2 = self.pool2(e22)

#         e31 = self.enc31(p2)
#         e32 = self.enc32(e31)
#         p3 = self.pool3(e32)

#         # Bottleneck
#         b1 = self.botteneck1(p3)
#         b2 = self.botteneck2(b1)

#         # Decoder foward pass with skip connections
#         d1 = self.up1(b2)
#         d1 = torch.cat([e32, d1], dim=1)  # TODO check if dim is correct
#         d1 = self.dec11(d1)
#         d1 = self.dec12(d1)

#         d2 = self.up2(d1)
#         d2 = torch.cat([e22, d2], dim=1)
#         d2 = self.dec21(d2)
#         d2 = self.dec22(d2)

#         d3 = self.up3(d2)
#         d3 = torch.cat([e12, d3], dim=1)
#         d3 = self.dec31(d3)
#         d3 = self.dec32(d3)

#         out = self.final(d3)
#         return out

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()

        # Encoder
        self.enc11 = NormConv3D(1, 16, kernel_size=3, stride=1, padding=1)
        self.enc12 = NormConv3D(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc21 = NormConv3D(32, 32, kernel_size=3, stride=1, padding=1)
        self.enc22 = NormConv3D(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.enc31 = NormConv3D(64, 64, kernel_size=3, stride=1, padding=1)
        self.enc32 = NormConv3D(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Bottleneck
        self.botteneck1 = NormConv3D(128, 128, kernel_size=3, stride=1, padding=1)
        self.botteneck2 = NormConv3D(128, 256, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.up1 = NormConvTranspose3D(256, 256, kernel_size=2, stride=2)
        self.dec11 = NormConv3D(384, 128, kernel_size=3, stride=1, padding=1)
        self.dec12 = NormConv3D(128, 128, kernel_size=3, stride=1, padding=1)

        self.up2 = NormConvTranspose3D(128, 128, kernel_size=2, stride=2)
        self.dec21 = NormConv3D(192, 64, kernel_size=3, stride=1, padding=1)
        self.dec22 = NormConv3D(64, 64, kernel_size=3, stride=1, padding=1)

        self.up3 = NormConvTranspose3D(64, 64, kernel_size=2, stride=2)
        self.dec31 = NormConv3D(96, 32, kernel_size=3, stride=1, padding=1)
        self.dec32 = NormConv3D(32, 32, kernel_size=3, stride=1, padding=1)

        self.final = nn.Conv3d(32, 6, kernel_size=1)  # TODO check if this is right

    def forward(self, x):
        # Encocder foward pass
        e11 = self.enc11(x)
        e12 = self.enc12(e11)
        p1 = self.pool1(e12)

        e21 = self.enc21(p1)
        e22 = self.enc22(e21)
        p2 = self.pool2(e22)

        e31 = self.enc31(p2)
        e32 = self.enc32(e31)
        p3 = self.pool3(e32)

        # Bottleneck
        b1 = self.botteneck1(p3)
        b2 = self.botteneck2(b1)

        # Decoder foward pass with skip connections
        d1 = self.up1(b2)
        d1 = torch.cat([e32, d1], dim=1)  # TODO check if dim is correct
        d1 = self.dec11(d1)
        d1 = self.dec12(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([e22, d2], dim=1)
        d2 = self.dec21(d2)
        d2 = self.dec22(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([e12, d3], dim=1)
        d3 = self.dec31(d3)
        d3 = self.dec32(d3)

        out = self.final(d3)
        return out

