"""
modules.py

Source code of the components of the vision transformer.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalFilterLayer(nn.Module):
    def __init__(self):
        super(GlobalFilterLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=(512, 16, 16))  # Assuming input size after patch embedding
        self.learnable_filter = nn.Parameter(torch.randn(512, 16, 16, dtype=torch.float32))  # Learnable filters in spatial domain

    def forward(self, x):
        x = self.layer_norm(x) # Normalize the input
        x_fft = torch.fft.fft2(x) # Apply 2D FFT
        filter_fft = torch.fft.fft2(self.learnable_filter) # Create a complex representation of the learnable filter
        x_fft_filtered = x_fft * filter_fft # Apply the learnable global filter in frequency domain
        x = torch.fft.ifft2(x_fft_filtered) # Apply 2D IFFT
        return x.abs()  # Return the magnitude after IFFT

class GFNet(nn.Module):
    def __init__(self, num_classes):
        super(GFNet, self).__init__()

        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=8, stride=8)  

        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.global_filter1 = GlobalFilterLayer()
        self.global_filter2 = GlobalFilterLayer()

        # Dropout layer to reduce overfitting
        self.dropout = nn.Dropout(p=0.6) 

        self.ffn = nn.Sequential(
            nn.LayerNorm(512 * 16 * 16),
            nn.Linear(512 * 16 * 16, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.classifier = nn.Linear(512, num_classes)     # Linear layer for classification

    def forward(self, x):
        x = self.patch_embedding(x)         # Patch embedding

        x = self.bn1(self.conv1(x))         # Convolutional layers
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        x = self.bn4(self.conv4(x))
        x = F.relu(x)

        x = self.global_filter1(x)          # Global Filter Layer 1
        x = self.global_filter2(x)          # Global Filter Layer 2
        x = x.view(x.size(0), -1)           # Flatten the output for the feedforward network
        x = self.ffn(x)                     # Feed Forward Network
        x =  x.view(x.size(0), 512, 1, 1)   # Reshape for global average pooling        
        x = self.global_avg_pooling(x)      # Global Average Pooling
        x = x.view(x.size(0), -1)           # Flatten again for classifier

        x = self.dropout(x)                 # Dropout before classifier

        return self.classifier(x)
