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


class Encoder(nn.Module):
    """Encoder for the VQVAE model."""
    def __init__(self, input_channels, hidden_channels, num_res_layers):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_res_layers)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder for the VQVAE model."""
    def __init__(self, hidden_channels, output_channels, num_res_layers):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_res_layers)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv2(x)
        return x
