import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(z_dim , w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

        def forward(self, z):
            return self.mapping(z)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def foward(self, image, noise):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise

class AdaIN(nn.Module):
    def __init__(self, in_channel, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(w_dim, in_channel * 2)

    def forward(self, image, w):
        style = self.style(w).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, dim=1)
        return (1 + gamma) * self.norm(image) + beta

class StyleConv(nn.Module):
    def __init(self, in_channel, out_channel, kernel_size, style_dim, upsample=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size//2)
        self.noise = NoiseInjection(out_channel)
        self.adain = AdaIN(out_channel, style_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.upsample = upsample

    def foward(self, input, style, noise=None):
        if self.upsample:
            input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        output = self.conv(input)
        output = self.noise(output, noise)
        output = self.adain(output, style)
        return self.activation(output)