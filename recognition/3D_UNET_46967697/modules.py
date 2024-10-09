import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_dim, out_dim)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_dim, out_dim)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(self, in_dim, num_classes, num_filters):
        super(Unet3D, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_filters = num_filters

        # Down sampling
        self.input = DoubleConv(in_dim, num_filters)

        self.down_1 = Down(num_filters, num_filters*2)
        self.down_2 = Down(num_filters*2, num_filters*4)
        self.down_3 = Down(num_filters*4, num_filters*8)
        self.down_4 = Down(num_filters*8, num_filters*16)

        # Up sampling
        self.up_1 = Up(num_filters*16, num_filters*8)
        self.up_2 = Up(num_filters*8, num_filters*4)
        self.up_3 = Up(num_filters*4, num_filters*2)
        self.up_4 = Up(num_filters*2, num_filters)

        self.output = nn.Conv3d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # Down sampling
        x1 = self.input(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        # Up sampling
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)

        # Output layer
        return self.output(x)
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        
        dice_coefficient = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_coefficient
