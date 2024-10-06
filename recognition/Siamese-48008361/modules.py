## Initial Comment

import torch
import torch.nn as nn
import torch.nn.functional as F

# Making our custom ResNet inspired by Resnet-18
# Residual block of ResNet
class ResBlock(nn.module):
    expansion = 1

    # Constructor for the ResBlock
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # Just two layers of Convolution and Batch Normalization
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        # If the input and output channels are not the same, then we need to adjust the shortcut connection
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Making the ResNet class   
class CustomResNet(nn.module):
    
