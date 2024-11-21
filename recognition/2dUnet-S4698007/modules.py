import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=6):  # Set default out_channels to 6
        super(UNet, self).__init__()
        
        # Encoder Path: This part reduces the spatial size of the image while increasing feature depth
        
        # The first layer of the encoder: takes the input and produces 32 feature maps
        self.encoder1 = self.conv_block(in_channels, 32) 
        
        # The second encoder layer: processes 32 feature maps and outputs 64 feature maps
        self.encoder2 = self.conv_block(32, 64)
        
        # The third encoder layer: takes 64 feature maps and outputs 128 feature maps
        self.encoder3 = self.conv_block(64, 128)
        
        # Bottleneck: The deepest layer in the network, which outputs 256 feature maps
        self.bottleneck = self.conv_block(128, 256)  
        
        # Decoder Path: This part restores the spatial size of the image step by step
        
        # Upsample from 256 to 128 feature maps (first step of upscaling)
        self.upconv3 = self.upconv(256, 128)
        
        # Concatenate with the corresponding encoder output and apply convolutional layers to merge the information
        self.decoder3 = self.conv_block(128 + 128, 128)
        
        # Upsample from 128 to 64 feature maps
        self.upconv2 = self.upconv(128, 64)
        
        # Concatenate with the corresponding encoder output and apply convolutional layers
        self.decoder2 = self.conv_block(64 + 64, 64)
        
        # Upsample from 64 to 32 feature maps
        self.upconv1 = self.upconv(64, 32)
        
        # Concatenate with the corresponding encoder output and apply convolutional layers
        self.decoder1 = self.conv_block(32 + 32, 32)
        
        # Final output layer: A 1x1 convolution that reduces the output to the desired number of channels (6 in this case)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        
        # Decoder path
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final output
        return self.final_conv(dec1)
