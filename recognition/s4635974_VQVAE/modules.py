import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_residual_hiddens, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, 
                               kernel_size=4, stride=2, padding=1)  # Strided convolution
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 
                               kernel_size=4, stride=2, padding=1)  # Strided convolution
        self.res_block1 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.res_block2 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x



