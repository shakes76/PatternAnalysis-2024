import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (downsampling) path
        self.down1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        
        # Decoder (upsampling) path
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

        # Activation
        self.relu = nn.ReLU()

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        # Encoder
        x1 = self.relu(self.bn1(self.down1(x)))  # Output size: 64 x 32 x 32
        x2 = self.relu(self.bn2(self.down2(x1))) # Output size: 128 x 16 x 16
        x3 = self.relu(self.bn3(self.down3(x2))) # Output size: 256 x 8 x 8
        x4 = self.relu(self.bn4(self.down4(x3))) # Output size: 512 x 4 x 4

        # Decoder with skip connections at the second and fourth layers
        x = self.relu(self.up1(x4))              # Output size: 256 x 8 x 8
        x = self.relu(self.up2(x + x4))          # Skip connection: add x4, Output size: 128 x 16 x 16
        x = self.relu(self.up3(x + x2))          # Skip connection: add x2, Output size: 64 x 32 x 32
        x = self.up4(x)                          # Final upsample to original size, Output size: 3 x 64 x 64

        return x
