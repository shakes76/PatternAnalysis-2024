import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    #3 input and output channels
    def __init__(self, in_channels=3, out_channels=3, base_filters=64):
        super(UNet3D, self).__init__()
        
        #Encoder definitions - downsampling
        self.encoder1 = self.conv_step(in_channels, base_filters)       
        self.encoder2 = self.conv_step(base_filters, base_filters*2)
        self.encoder3 = self.conv_step(base_filters*2, base_filters*4)
        self.encoder4 = self.conv_step(base_filters*4, base_filters*8)

        #Decoder definitons - upsampling
        self.decoder4 = self.conv_step(base_filters*16, base_filters*8)
        self.decoder3 = self.conv_step(base_filters*8, base_filters*4)
        self.decoder2 = self.conv_step(base_filters*4, base_filters*2)
        self.decoder1 = self.conv_step(base_filters*2, base_filters)
        
        
        
        #Upconvolutions
        self.upconv4 = self.upconv(base_filters*16, base_filters*8)
        self.upconv3 = self.upconv(base_filters*8, base_filters*4)
        self.upconv2 = self.upconv(base_filters*4, base_filters*2)
        self.upconv1 = self.upconv(base_filters*2, base_filters)

        #final convolution step
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)

        #Bridge convolution between encoder and decoder
        self.bridge = self.conv_step(base_filters*8, base_filters*16)

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
