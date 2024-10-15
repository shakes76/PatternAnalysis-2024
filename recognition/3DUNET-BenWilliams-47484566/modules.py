import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    #3 input and output channels
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNet3D, self).__init__()
    
    
    



    def conv_step(self, in_channels, out_channels):
        #conv + BN + relu step between feature maps
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv(self, in_channels, out_channels):
        # Transposed convolution for upsampling
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
