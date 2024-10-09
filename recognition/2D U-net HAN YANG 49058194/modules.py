import torch
import torch.nn as nn

# Two layers of convolutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    # Forward propagation
    def forward(self, x):
        return self.conv(x)
        
# Downsampling operation of U-Net
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    # Forward propagation
    def forward(self, x):
        return self.maxpool_conv(x)

# Upsampling operation of U-Net
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    # Forward propagation
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure x1 and x2 are the same size by cropping x2
        if x1.size()[2:] != x2.size()[2:]:
            x2 = self._crop_tensor(x2, x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    # Upsampling and downsampling feature concatenation
    def _crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2:]
        tensor_size = tensor.size()[2:]
        delta_h = tensor_size[0] - target_size[0]
        delta_w = tensor_size[1] - target_size[1]
        return tensor[:, :, delta_h // 2:tensor_size[0] - delta_h // 2, delta_w // 2:tensor_size[1] - delta_w // 2]

# Out convolution
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    # Forward propagation
    def forward(self, x):
        return self.conv(x)

# U-net model
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
