import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DoubleConv(nn.Module):
    """
    Applies two consecutive convolutional layers with ReLU activation and optional batch normalisation.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, use_batchnorm: bool = True):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ])
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
    
class Down(nn.Module):
    """
    Downscaling with maxpool followed by double convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling with transposed convolution or bilinear upsampling followed by double convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, use_batchnorm: bool = True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
            factor = 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            factor = 1
        self.conv = DoubleConv(in_channels, out_channels, use_batchnorm=use_batchnorm)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        # Padding to handle odd input dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    """2D UNet architecture with configurable parameters."""
    def __init__(self, 
                 n_channels: int = 1, 
                 n_classes: int = 1, 
                 bilinear: bool = True,
                 base_filters: int = 64,
                 use_batchnorm: bool = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, base_filters, use_batchnorm=use_batchnorm)
        self.down1 = Down(base_filters, base_filters * 2, use_batchnorm=use_batchnorm)
        self.down2 = Down(base_filters * 2, base_filters * 4, use_batchnorm=use_batchnorm)
        self.down3 = Down(base_filters * 4, base_filters * 8, use_batchnorm=use_batchnorm)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_filters * 8, base_filters * 16 // factor, use_batchnorm=use_batchnorm)
        self.up1 = Up(base_filters * 16, base_filters * 8 // factor, bilinear, use_batchnorm=use_batchnorm)
        self.up2 = Up(base_filters * 8, base_filters * 4 // factor, bilinear, use_batchnorm=use_batchnorm)
        self.up3 = Up(base_filters * 4, base_filters * 2 // factor, bilinear, use_batchnorm=use_batchnorm)
        self.up4 = Up(base_filters * 2, base_filters, bilinear, use_batchnorm=use_batchnorm)
        self.outc = nn.Conv2d(base_filters, n_classes, kernel_size=1)
        
        # Initialise weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # base_filters
        x2 = self.down1(x1)    # base_filters * 2
        x3 = self.down2(x2)    # base_filters * 4
        x4 = self.down3(x3)    # base_filters * 8
        x5 = self.down4(x4)    # base_filters * 16
        
        x = self.up1(x5, x4)   # base_filters * 8
        x = self.up2(x, x3)    # base_filters * 4
        x = self.up3(x, x2)    # base_filters * 2
        x = self.up4(x, x1)    # base_filters
        logits = self.outc(x)
        return logits
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=1)
    x = torch.randn(1, 1, 256, 256)  # Example input tensor
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")