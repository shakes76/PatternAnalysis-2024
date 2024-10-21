import torch
import torch.nn as nn
import numpy as np

NEGATIVE_SLOPE = 10 ** -2
DROP_PROB = 0.3
NUM_SEGMENTS = 6

class StandardModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 3, stride = 1, padding = 1, inplace = False):
        super(StandardModule, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                                kernel_size = kernel_size, stride = stride, padding = padding)
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.l_relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace = inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.l_relu(x)
        return x
    
class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size = 3, stride = 1, padding = 1):
        super(ContextModule, self).__init__()
        self.block1 = StandardModule(in_channels = in_channels, out_channels = out_channels, inplace = True)
        self.dropout = nn.Dropout(DROP_PROB) # Drop description uncertain, replace it with Dropout3d when poor performance
        self.block2 = StandardModule(in_channels = in_channels, out_channels = out_channels, inplace = True)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.dropout(x)
        x = self.block2(x)
        return x
    
class Stride2Module(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=2, padding=1, inplace = False):
        super(Stride2Module, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                                kernel_size = kernel_size, stride = stride, padding = padding)
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.l_relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace = inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.l_relu(x)
        return x
    
class UpsamplingModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingModule, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels,
                                                 kernel_size = 4, stride = 2, padding = 1)
        self.block = StandardModule(in_channels = out_channels, out_channels = out_channels, inplace = True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.block(x)
        return x


class LocalisationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalisationModule, self).__init__()

        self.block1 = StandardModule(in_channels = in_channels, out_channels = out_channels)

        self.block2 = StandardModule(in_channels = out_channels, out_channels = out_channels, kernel_size = 1, padding = 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x
    
class SegmentationLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationLayer, self).__init__()
        self.seg = nn.Conv3d(in_channels = in_channels, out_channels = out_channels,
                                      kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        return self.seg(x)
    
class UpScaleModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpScaleModule, self).__init__()
        self.upscale = nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels,
                                          kernel_size = 4, stride = 2, padding = 1)

    def forward(self, x):
        return self.upscale(x)
    
class ImprovedUnet(nn.Module):
    def __init__(self):
        super(ImprovedUnet, self).__init__()
        self.block1 = StandardModule(1, 16) # Grayscale thus requries 1 input channel
        self.context1 = ContextModule(16, 16)

        self.block2 = Stride2Module(16, 32)
        self.context2 = ContextModule(32, 32)

        self.block3 = Stride2Module(32, 64)
        self.context3 = ContextModule(64, 64)

        self.block4 = Stride2Module(64, 128)
        self.context4 = ContextModule(128, 128)

        self.block5 = Stride2Module(128, 256)
        self.context5 = ContextModule(256, 256)        

        self.upsample1 = UpsamplingModule(256, 128)

        self.localise1 = LocalisationModule(256, 128)
        self.upsample2 = UpsamplingModule(128, 64)

        self.localise2 = LocalisationModule(128, 64)
        self.upsample3 = UpsamplingModule(64, 32)

        self.localise3 = LocalisationModule(64, 32)
        self.upsample4 = UpsamplingModule(32, 16)

        self.conv_output = StandardModule(32, 32)

        # first segmentation layer
        self.segmentation1 = SegmentationLayer(64, NUM_SEGMENTS)

        # second segmentation layer
        self.segmentation2 = SegmentationLayer(32, NUM_SEGMENTS)

        # third segmentation layer
        self.segmentation3 = SegmentationLayer(32, NUM_SEGMENTS)

        # upscaling layers
        self.upscale_1 = UpScaleModule(NUM_SEGMENTS, NUM_SEGMENTS)
        self.upscale_2 = UpScaleModule(NUM_SEGMENTS, NUM_SEGMENTS)


    def forward(self, x):
        conv_out_1 = self.block1(x)
        context_out_1 = self.context1(conv_out_1)
        element_sum_1 = conv_out_1 + context_out_1
        # second term
        conv_out_2 = self.block2(element_sum_1)
        context_out_2  = self.context2(conv_out_2)
        element_sum_2 = conv_out_2 + context_out_2

        # third downsample
        conv_out_3 = self.block3(element_sum_2)
        context_out_3 = self.context3(conv_out_3)
        element_sum_3 = conv_out_3 + context_out_3

        conv_out_4 = self.block4(element_sum_3)
        context_out_4 = self.context4(conv_out_4)
        element_sum_4 = conv_out_4 + context_out_4

        conv_out_5 = self.block5(element_sum_4)
        context_out_5 = self.context5(conv_out_5)
        element_sum_5 = conv_out_5 + context_out_5

        # First upsampling module.
        upsample_out_1 = self.upsample1(element_sum_5)
        concat_1 = torch.cat((element_sum_4, upsample_out_1), dim = 1)

        localisation_out_1 = self.localise1(concat_1)
        upsample_out_2 = self.upsample2(localisation_out_1)
        concat_2 = torch.cat((element_sum_3, upsample_out_2), dim = 1)

        localisation_out_2 = self.localise2(concat_2)
        upsample_out_3 = self.upsample3(localisation_out_2)
        concat_3 = torch.cat((element_sum_2, upsample_out_3), dim = 1)

        localisation_out_3 = self.localise3(concat_3)
        upsample_out_4 = self.upsample4(localisation_out_3)
        concat_4 = torch.cat((element_sum_1, upsample_out_4), dim = 1)

        segment_out_1 = self.segmentation1(localisation_out_2)
        upscale_out_1 = self.upscale_1(segment_out_1)

        segment_out_2 = self.segmentation2(localisation_out_3)
        seg_sum_1 = upscale_out_1 + segment_out_2
        
        upscale_out_2 = self.upscale_2(seg_sum_1)
        convoutput_out = self.conv_output(concat_4)
        segment_out_3 = self.segmentation3(convoutput_out)

        final_sum = upscale_out_2 + segment_out_3
        
        output =  torch.softmax(final_sum, dim = 1)
        return output