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

class Generator(nn.Module):
    def __init__(self, style_dim, num_layers, channels):
        super().__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.channels = channels
        self.input = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.conv1 = StyleConv(channels[0], channels[0], 3, style_dim, upsample=True)
        self.to_rgb1 = nn.Conv2d(channels[0], 3, 1)

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        for i in range(1, num_layers):
            self.convs.append(StyleConv(channels[i-1], channels[i], 3, style_dim, upsample=True))
            self.to_rgbs.append(nn.Conv2d(channels[i], 3, 1))

    def foward(self, styles, noise = None):
        if noise is None:
            noise = [None] * self.num_layers

        out = self.input.repeat(styles.shape[0], 1, 1, 1)
        out = self.conv1(out, styles[:, 0], noise[0])

        skip = self.to_rgb1(out)

        for i in range(1, self.num_layers):
            out = self.convs[i](out, styles[:, i], noise[i])
            skip = skip + self.to_rgbs[i](out)

        image = skip / self.num_layers
        return image

class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels[::-1]

        layers = []
        for i in range(len(self.channels) - 1):
            layers.append(nn.Conv2d(self.channels[i], self.channels[i+1], 3, padding=1))
            layers.append(nn.LeakyReLU(0.2))

        self.convs = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(self.channels[-1], 1, 4)

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)
        return out.view(out.shape[0], -1)

class StyleGAN2(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers, channels):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim, 8)
        self.generator = Generator(w_dim, num_layers, channels)
        self.discriminator = Discriminator(channels[::-1])

    def generate(self, z):
        styles = self.mapping(z)
        return self.generator(styles)

    def discriminate(self, image):
        return self.discriminator(image)
