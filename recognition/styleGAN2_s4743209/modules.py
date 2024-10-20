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

    def forward(self, image, noise):
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
        # Ensure w is 2D: (batch_size, w_dim)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        style = self.style(w).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, dim=1)
        out = self.norm(image)
        return (1 + gamma) * out + beta
class StyleConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2)
        self.noise = NoiseInjection(out_channel)
        self.adain = AdaIN(out_channel, style_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.upsample = upsample

    def forward(self, input, style, noise=None):
        if self.upsample:
            input = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv(input)
        out = self.noise(out, noise=noise)
        out = self.adain(out, style)
        return self.activation(out)


class Generator(nn.Module):
    def __init__(self, style_dim, num_layers, channels, img_size=256):
        super().__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.channels = channels
        self.img_size = img_size

        self.input = nn.Parameter(torch.randn(1, channels[0], 4, 4))
        self.conv1 = StyleConv(channels[0], channels[0], 3, style_dim, upsample=True)
        self.to_rgb1 = nn.Conv2d(channels[0], 1, 1)  # Changed to 1 channel output

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        in_channel = channels[0]
        for i in range(1, num_layers):
            out_channel = channels[i]
            self.convs.append(StyleConv(in_channel, out_channel, 3, style_dim, upsample=False))
            self.to_rgbs.append(nn.Conv2d(out_channel, 1, 1))  # Changed to 1 channel output
            self.upsamples.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            in_channel = out_channel

    def forward(self, styles, noise=None):
        if noise is None:
            noise = [None] * self.num_layers

        if styles.dim() == 2:
            styles = styles.unsqueeze(1).repeat(1, self.num_layers, 1)

        batch_size = styles.size(0)
        out = self.input.repeat(batch_size, 1, 1, 1)
        out = self.conv1(out, styles[:, 0], noise[0])

        skip = self.to_rgb1(out)

        for i in range(1, self.num_layers):
            out = self.convs[i-1](out, styles[:, i], noise[i])
            out = self.upsamples[i-1](out)
            skip = self.upsamples[i-1](skip)
            skip = skip + self.to_rgbs[i-1](out)

        image = skip

        if image.size(-1) != self.img_size:
            image = F.interpolate(image, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return image

class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        layers = []
        in_channel = 1  # Start with 1 channel for grayscale images
        for out_channel in channels:
            layers.append(nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.2))
            in_channel = out_channel

        self.convs = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(channels[-1], 1, 2)

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)
        return out.view(out.shape[0], -1)

class StyleGAN2(nn.Module):
    def __init__(self, w_dim, num_layers, channels, img_size=256):
        super().__init__()
        self.mapping = MappingNetwork(w_dim, w_dim, 8)
        self.generator = Generator(w_dim, num_layers, channels, img_size)
        self.discriminator = Discriminator(channels)

    def generate(self, z):
        styles = self.mapping(z)
        return self.generator(styles)

    def discriminate(self, image):
        return self.discriminator(image)