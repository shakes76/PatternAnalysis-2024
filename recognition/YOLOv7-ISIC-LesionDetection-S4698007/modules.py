import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder Path: This part reduces the spatial size of the image while increasing feature depth
        
        # The first layer of the encoder: takes the input and produces 32 feature maps
        self.encoder1 = self.conv_block(in_channels, 32) 
        
        # The second encoder layer: processes 32 feature maps and outputs 64 feature maps
        self.encoder2 = self.conv_block(32, 64)
        
        # The third encoder layer: takes 64 feature maps and outputs 128 feature maps
        self.encoder3 = self.conv_block(64, 128)
        
        # Bottleneck: The deepest layer in the network, which outputs 256 feature maps
        # This is where the image is reduced the most in spatial size but represented with the most depth
        self.bottleneck = self.conv_block(128, 256)  
        
        # Decoder Path: This part restores the spatial size of the image step by step
        
        # Upsample from 256 to 128 feature maps (first step of upscaling)
        self.upconv3 = self.upconv(256, 128)
        
        # After upsampling, concatenate with the corresponding encoder output (128 + 128 channels) 
        # and then apply convolutional layers to merge the information
        self.decoder3 = self.conv_block(128 + 128, 128)
        
        # Upsample from 128 to 64 feature maps
        self.upconv2 = self.upconv(128, 64)
        
        # Concatenate with the corresponding encoder output (64 + 64 channels)
        # and apply convolutional layers to refine the information
        self.decoder2 = self.conv_block(64 + 64, 64)
        
        # Upsample from 64 to 32 feature maps
        self.upconv1 = self.upconv(64, 32)
        
        # Concatenate with the corresponding encoder output (32 + 32 channels)
        # and apply convolutional layers to complete the final part of the decoder
        self.decoder1 = self.conv_block(32 + 32, 32)
        
        # Final output layer: A 1x1 convolution that reduces the output to the desired number of channels
        # (e.g., 1 for binary segmentation, where each pixel is classified as either background or object)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        # This function defines a basic block of two convolutional layers, each followed by
        # batch normalization and a ReLU activation to introduce non-linearity.
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def upconv(self, in_channels, out_channels):
        # This function defines a transposed convolution (also known as upsampling) to double the spatial size
        # of the feature map while reducing the number of channels
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # The forward function defines how the data flows through the model.
        
        # Pass the input through the first encoder block
        enc1 = self.encoder1(x)
        
        # Downsample using max pooling, then pass through the second encoder block
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        
        # Downsample again and pass through the third encoder block
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        
        # Downsample one more time and pass through the bottleneck layer (deepest part of the network)
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        
        # Now, we start upsampling in the decoder path.
        
        # Upsample the bottleneck output and concatenate with the corresponding encoder output
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        # Upsample again and concatenate with the next encoder output
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        # Upsample one more time and concatenate with the first encoder output
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Pass the final decoder output through the last 1x1 convolution to get the final result
        return self.final_conv(dec1)
