# Containing the source code of the components of your model. 
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLayer(nn.Module):
    """A basic residual layer."""
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        return out


class ResidualStack(nn.Module):
    """Stack of residual layers."""
    def __init__(self, in_channels, out_channels, num_layers):
        super(ResidualStack, self).__init__()
        layers = [ResidualLayer(in_channels, out_channels) for _ in range(num_layers)]
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)



