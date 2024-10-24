import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder path
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder path
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(512 + 512, 512)   # Skip connection with encoder4
        
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(256 + 256, 256)   # Skip connection with encoder3
        
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(128 + 128, 128)   # Skip connection with encoder2
        
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(64 + 64, 64)      # Skip connection with encoder1
        
        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        # A convolutional block consisting of two convolutional layers, batch normalization, and ReLU activation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)   # Add dropout after the second conv layer as well
        )
    
    def upconv(self, in_channels, out_channels):
        # A transposed convolution layer for upsampling
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # Concatenating encoder feature maps
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenating encoder feature maps
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenating encoder feature maps
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenating encoder feature maps
        dec1 = self.decoder1(dec1)
        
        # Final output layer
        return self.final_conv(dec1)

# Example usage:
# model = ComplexUNet(in_channels=1, out_channels=1)
# output = model(input_tensor)
