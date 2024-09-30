import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.mapping(x)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(w_dim, channels)
        self.style_bias = nn.Linear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).view(-1, x.size(1), 1, 1)
        style_bias = self.style_bias(w).view(-1, x.size(1), 1, 1)
        return style_scale * x + style_bias

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super().__init__()
        self.starting_cte = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain = AdaIN(in_channels, w_dim)
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.rgb_layer = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, noise):
        w = self.map(noise)
        x = self.initial_adain(self.starting_cte, w)
        x = self.initial_conv(x)
        return self.rgb_layer(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.final_block = nn.Sequential(
            WSConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.initial_rgb(x)
        return self.final_block(out).view(out.shape[0], -1)

