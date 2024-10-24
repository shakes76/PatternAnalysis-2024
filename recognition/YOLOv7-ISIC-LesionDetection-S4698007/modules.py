import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder path
        self.encoder1 = self.conv_block(in_channels, 32)  # Reduced to 32
        self.encoder2 = self.conv_block(32, 64)            # Reduced to 64
        self.encoder3 = self.conv_block(64, 128)           # Reduced to 128
        self.bottleneck = self.conv_block(128, 256)        # Reduced to 256
        
        # Decoder path
        self.upconv3 = self.upconv(256, 128)
        self.decoder3 = self.conv_block(128 + 128, 128)   # After concatenation
        
        self.upconv2 = self.upconv(128, 64)
        self.decoder2 = self.conv_block(64 + 64, 64)       # After concatenation
        
        self.upconv1 = self.upconv(64, 32)
        self.decoder1 = self.conv_block(32 + 32, 32)       # After concatenation
        
        # Final output
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
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc3, 2))
        
        # Decoder path with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)  # Concatenating encoder feature maps
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Concatenating encoder feature maps
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # Concatenating encoder feature maps
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)