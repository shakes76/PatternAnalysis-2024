"""
Contains the source code for the components of GFNet classifying the Alzheimer’s disease (normal and AD) of the ADNI brain data
Each component is implementated as a class or a function.
"""

import torch
import torch.nn as nn
import torch.fft

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x

class GlobalFilterNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Global Filter layers
        self.global_filter1 = GlobalFilter(dim=64)
        self.global_filter2 = GlobalFilter(dim=128)

        # Additional convolution layers
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        - Pass the input through the initial convolutional blocks
        - Reshape the feature maps to prepare for the global filtering layers
        - Apply the two global filtering layers
        - Pass the filtered features through the final classification layers (average pooling and fully connected)
        - Return the final classification output
        """
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Reshape for GlobalFilter
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.global_filter1(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Second GlobalFilter
        x = x.permute(0, 2, 3, 1)
        x = self.global_filter2(x)
        x = x.permute(0, 3, 1, 2)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x