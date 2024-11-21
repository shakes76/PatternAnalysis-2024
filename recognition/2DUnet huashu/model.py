import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet2D, self).__init__()
        self.encoder1 = self.contract_block(in_channels, 64, 3, 1) 
        self.encoder2 = self.contract_block(64, 128, 3, 1)
        self.encoder3 = self.contract_block(128, 256, 3, 1)
        self.encoder4 = self.contract_block(256, 512, 3, 1)
        
        self.middle = self.contract_block(512, 1024, 3, 1)
        
        self.upconv4 = self.expand_block(1024, 512, 3, 1)
        self.upconv3 = self.expand_block(512, 256, 3, 1)
        self.upconv2 = self.expand_block(256, 128, 3, 1)
        self.upconv1 = self.expand_block(128, 64, 3, 1)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        middle = self.middle(F.max_pool2d(enc4, 2))

        # Adjust the number of lanes for enc4 to 512 to match the number of channels in the middle
        enc4_adjusted = nn.Conv2d(512, 512, kernel_size=1)(enc4)

        # Splice the middle and enc4_adjusted, and adjust the number of channels
        concatenated = torch.cat([F.interpolate(middle, scale_factor=2), enc4_adjusted], 1)
        concatenated_adjusted = nn.Conv2d(1536, 1024, kernel_size=1)(concatenated)

        # Upsampling and decoder sections
        dec4 = self.upconv4(concatenated_adjusted)

        # Splice dec4 and enc3 and adjust the number of channels
        dec4_enc3 = torch.cat([F.interpolate(dec4, scale_factor=2), enc3], 1)
        dec4_enc3_adjusted = nn.Conv2d(768, 512, kernel_size=1)(dec4_enc3)

        dec3 = self.upconv3(dec4_enc3_adjusted)

        # Splice dec3 and enc2 and adjust the number of channels
        dec3_enc2 = torch.cat([F.interpolate(dec3, scale_factor=2), enc2], 1)
        dec3_enc2_adjusted = nn.Conv2d(384, 256, kernel_size=1)(dec3_enc2)

        dec2 = self.upconv2(dec3_enc2_adjusted)

        # Splice dec2 and enc1 and adjust the number of channels
        dec2_enc1 = torch.cat([F.interpolate(dec2, scale_factor=2), enc1], 1)
        dec2_enc1_adjusted = nn.Conv2d(192, 128, kernel_size=1)(dec2_enc1)  # Adjust the number of channels to 128

        dec1 = self.upconv1(dec2_enc1_adjusted)

        return torch.sigmoid(self.final_conv(dec1))

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block
