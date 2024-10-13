"""
containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""

# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global variables
NEGATIVE_SLOPE = 10**2
DROPOUT_PROB = 0.3

"""
Custom convolutional layer with adjusted padding to retain original spatial dimensions 
It includes an instance normalization and LeakyReLU activation.

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.
- kernel_size (int, optional): Size of the convolutional kernel (default is 3).
- stride (int, optional): Stride of the convolution operation (default is 1).
- padding (int, optional): Padding size for the convolution (default is 0).
"""
class StandardConv(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(StandardConv, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE)

    def forward(self, x):
        x = self.conv3d(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        return x

"""
Context module class for enhanced feature mapping using pre-activation residual blocks with dropout between.

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.

Architecture:
- Conv1: 3x3 convolutional layer with instance normalisation and LeakyReLU activation.
- Dropout: Dropout layer with a specified dropout probability.
- Conv2: 3x3 convolutional layer with instance normalisation and LeakyReLU activation.
"""
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ContextModule, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)
        self.dropout = nn.Dropout(DROPOUT_PROB)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.instance_norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)
        x = self.relu2(x)
        return x

"""
Convolutional Layer with Stride

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.
- kernel_size (int, optional): Size of the convolutional kernel (default is 3).
- stride (int, optional): Stride of the convolution operation (default is 2).
- padding (int, optional): Padding size for the convolution (default is 0). Adjusted padding to retain spatial dimensions.

"""
class ConvWithStride(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1):
        super(ConvWithStride, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

"""
Upsampling Module

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.

Architecture:
- ConvTranspose2d: Transposed convolution for upsampling with a kernel size of 2 and stride of 2.
- Conv: 3x3 convolutional layer with padding to maintain spatial dimensions.
- InstanceNorm: Instance normalisation applied to the output.
- LeakyReLU: Leaky ReLU activation function.
"""
class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                 kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv3d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


"""
Localisation Module for recombining features. 

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.

Architecture:
- Conv1: 3x3 convolutional layer with instance normalisation and LeakyReLU activation.
- Conv2: 1x1 convolutional layer with instance normalisation and LeakyReLU activation.
"""
class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, padding=0)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.relu2 = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


"""
Segmentation Module reduces depth of feature maps to 1. 

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels (set to 1 since just mask is grayscale)

Architecture:
- Segmentation: 1x1 convolutional layer

"""
class SegmentationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationModule, self).__init__()
        self.segmentation = nn.Conv3d(in_channels, out_channels,
                                      kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.segmentation(x)

"""
UpScale Module used to increase spatial resolution. 

Parameters:
- in_channels (int): Number of input channels.
- out_channels (int): Number of output channels.

Architecture:
- Upscale: 2x2 transposed convolutional layer with no padding

"""
class UpScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpScaleModule, self).__init__()
        self.upscale = nn.ConvTranspose3d(in_channels, out_channels,
                                          kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.upscale(x)

"""
This class defines an improved U-Net architecture. 
"""
class ImprovedUnet(nn.Module):
    def __init__(self):
        super(ImprovedUnet, self).__init__()
        # in_channels = 3 # RGB colours have three input channels.
        in_channels = 1

        out_channels = 16
        self.conv_1 = StandardConv(in_channels, out_channels)
        in_channels=16
        self.context_1 = ContextModule(in_channels, out_channels)

        in_channels = 16
        out_channels = 32
        self.conv_2 = ConvWithStride(in_channels, out_channels)
        in_channels = 32
        self.context_layer_2 = ContextModule(in_channels, out_channels)

        in_channels = 32
        out_channels = 64
        self.conv_3 = ConvWithStride(in_channels, out_channels)
        in_channels = 64
        self.context_layer_3 = ContextModule(in_channels, out_channels)

        in_channels = 64
        out_channels = 128
        self.conv_4 = ConvWithStride(in_channels, out_channels)
        in_channels = 128
        self.context_layer_4 = ContextModule(in_channels, out_channels)

        in_channels = 128
        out_channels = 256
        self.conv_5 = ConvWithStride(in_channels, out_channels)
        in_channels = 256
        self.context_layer_5 = ContextModule(in_channels, out_channels)

        in_channels = 256
        out_channels = 128
        self.upsample_layer_1 = UpsamplingModule(in_channels, out_channels)

        in_channels = 256
        out_channels = 128
        self.localise_layer_1 = LocalisationModule(in_channels,out_channels)

        in_channels = 128
        out_channels = 64
        self.upsample_layer_2 = UpsamplingModule(in_channels, out_channels)

        in_channels = 128
        out_channels = 64
        self.localise_layer_2 = LocalisationModule(in_channels, out_channels)

        in_channels = 64
        out_channels = 32
        self.upsample_layer_3 = UpsamplingModule(in_channels, out_channels)

        in_channels = 64
        out_channels = 32
        self.localise_layer_3 = LocalisationModule(in_channels, out_channels)

        # fourth upsample layer
        in_channels = 32
        out_channels = 16
        self.upsample_layer_4 = UpsamplingModule(in_channels, out_channels)

        in_channels = 32
        out_channels = 32
        self.conv_output = StandardConv(in_channels, out_channels)

        # first segmentation layer
        in_channels =  64
        # out_channels = 1
        out_channels = 6
        self.segmentation_layer_1 = SegmentationModule(in_channels, out_channels)

        # second segmentation layer
        in_channels = 32
        # out_channels = 1
        out_channels = 6
        self.segmentation_layer_2 = SegmentationModule(in_channels, out_channels)

        # third segmentation layer
        in_channels = 32
        # out_channels = 1
        out_channels = 6
        self.segmentation_layer_3 = SegmentationModule(in_channels, out_channels)

        # upscaling layers
        self.upscale_1 = UpScaleModule(out_channels, out_channels)
        self.upscale_2 = UpScaleModule(out_channels, out_channels)


    def forward(self, x):
        conv_out_1 = self.conv_1(x)
        context_out_1 = self.context_1(conv_out_1)
        element_sum_1 = conv_out_1 + context_out_1

        # second term
        conv_out_2 = self.conv_2(element_sum_1)
        context_out_2  = self.context_layer_2(conv_out_2)
        element_sum_2 = conv_out_2 + context_out_2

        # third downsample
        conv_out_3 = self.conv_3(element_sum_2)
        context_out_3 = self.context_layer_3(conv_out_3)
        element_sum_3 = conv_out_3 + context_out_3

        conv_out_4 = self.conv_4(element_sum_3)
        context_out_4 = self.context_layer_4(conv_out_4)
        element_sum_4 = conv_out_4 + context_out_4

        conv_out_5 = self.conv_5(element_sum_4)
        context_out_5 = self.context_layer_5(conv_out_5)
        element_sum_5 = conv_out_5 + context_out_5

        # First upsampling module.
        upsample_out_1 = self.upsample_layer_1(element_sum_5)
        concat_1 = torch.cat((element_sum_4, upsample_out_1), dim=1)

        localisation_out_1 = self.localise_layer_1(concat_1)
        upsample_out_2 = self.upsample_layer_2(localisation_out_1)
        concat_2 = torch.cat((element_sum_3, upsample_out_2), dim=1)

        localisation_out_2 = self.localise_layer_2(concat_2)
        upsample_out_3 = self.upsample_layer_3(localisation_out_2)
        concat_3 = torch.cat((element_sum_2, upsample_out_3), dim=1)

        localisation_out_3 = self.localise_layer_3(concat_3)
        upsample_out_4 = self.upsample_layer_4(localisation_out_3)
        concat_4 = torch.cat((element_sum_1, upsample_out_4), dim=1)

        segment_out_1 = self.segmentation_layer_1(localisation_out_2)
        upscale_out_1 = self.upscale_1(segment_out_1)

        segment_out_2 = self.segmentation_layer_2(localisation_out_3)
        seg_sum_1 = upscale_out_1 + segment_out_2

        upscale_out_2 = self.upscale_2(seg_sum_1)

        segment_out_1 = self.segmentation_layer_1(localisation_out_2)
        upscale_out_1 = self.upscale_1(segment_out_1)

        convoutput_out = self.conv_output(concat_4)
        segment_out_3 = self.segmentation_layer_3(convoutput_out)

        final_sum = upscale_out_2 + segment_out_3

        output =  nn.Softmax(dim = 1)(final_sum)

        output = output.unsqueeze(1)

        output = output.permute(0, 1, 3, 4, 5, 2)

        return output