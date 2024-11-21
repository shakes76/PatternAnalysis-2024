"""
Contains custom PyTorch modules for StyleGAN2.

Acknowledgements:
Resources used to make the following modules:
    https://github.com/aburo8/PatternAnalysis-2023/tree/topic-recognition/recognition/46990480_StyleGAN2
    https://blog.paperspace.com/implementation-stylegan2-from-scratch/#models-implementation
    https://arxiv.org/abs/1812.04948
    https://arxiv.org/abs/1912.04958
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class EQWeight(nn.Module):
    """
    Equalised weight layer - normalises variance of initialised weights.
    """
    def __init__(self, shape):
        super(EQWeight, self).__init__()
        self.scale = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.scale

class EQLinearLayer(nn.Module):
    """
    Fully Connected Layer - Equalised Learning Rate
    """
    def __init__(self, in_dim, out_dim, bias=0.) -> None:
        super(EQLinearLayer, self).__init__()
        self.weight = EQWeight([out_dim, in_dim])
        self.bias = nn.Parameter(torch.ones(out_dim) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight().to(x.device), bias=self.bias.to(x.device))

class MappingNetwork(nn.Module):
    """
    Fully Connected Noise Mapping Network.
    """
    def __init__(self, z_dim, w_dim) -> None:
        super(MappingNetwork, self).__init__()
        self.mapping = nn.Sequential(
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(z_dim, w_dim)
        )

    def forward(self, x):
        # Pixel-wise normalisation for input
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(x)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation
    """
    def __init__(self, channels, w_dim) -> None:
        super(AdaIN, self).__init__()
        self.inst_norm = nn.InstanceNorm2d(channels)
        self.style_weight = EQLinearLayer(w_dim, channels) # scale of style
        self.style_bias = EQLinearLayer(w_dim, channels) # style shift

    def forward(self, x, w):
        # Instance Normalisation
        x = self.inst_norm(x)

        # Apply style transformation
        style_weight = self.style_weight(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)

        # Apply style to normalized input
        return x * style_weight + style_bias

class NoiseInjection(nn.Module):
    """
    Inject noise into the synthesis network.
    """
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        self.weight = self.weight.to(x.device)
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise

class Conv2dWeightModulate(nn.Module):
    '''
    Weight Modulation Convolutional Layer
    '''
    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EQWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps

    def forward(self, x, s):
        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)

class ToRGB(nn.Module):
    '''
    Generates an RGB image from a feature map using a 1x1 convolution
    '''

    def __init__(self, W_DIM, features):
        super().__init__()
        self.to_style = EQLinearLayer(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])

class StyleBlock(nn.Module):
    """
    Single Style Block For Synthesis Network
    """
    def __init__(self, w_dim, in_channels, out_channels, upsample=False) -> None:
        super(StyleBlock, self).__init__()
        self.upsample = upsample
        self.to_style = EQLinearLayer(w_dim, in_channels, bias=1.0)
        self.conv = Conv2dWeightModulate(in_channels, out_channels, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

class GeneratorBlock(nn.Module):
    """
    Generator Block - Comprised of Multiple Style Blocks
    """
    def __init__(self, w_dim, in_channels, out_channels, upsample=True) -> None:
        super(GeneratorBlock, self).__init__()

        # Style blocks
        self.style_block1 = StyleBlock(w_dim, in_channels, out_channels)
        self.style_block2 = StyleBlock(w_dim, out_channels, out_channels)

        self.to_rgb = ToRGB(w_dim, out_channels)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb

class Generator(nn.Module):
    """
    Generator Network - StyleGAN1.
    """
    def __init__(self, num_layers, w_dim, n_features=32, max_features=256) -> None:
        super(Generator, self).__init__()
        self.w_dim = w_dim
        self.num_layers = num_layers

        features = [min(max_features, n_features * (2 ** i)) for i in range(num_layers - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(w_dim, features[0], features[0])
        self.to_rgb = ToRGB(w_dim, features[0])

        blocks = [GeneratorBlock(w_dim, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):
        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)

class EQConv2d(nn.Module):
    """
    Conv2d with Equalised Learning Rate
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0) -> None:
        super(EQConv2d, self).__init__()
        self.padding = padding
        self.weight = EQWeight([out_channels, in_channels, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_channels))

    def forward(self, x):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block - ProGAN
    """
    def __init__(self, in_channels, out_channels, downsample=True) -> None:
        super(DiscriminatorBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
            EQConv2d(in_channels, out_channels, kernel_size=1)
        )

        self.block = nn.Sequential(
            EQConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EQConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        self.scale = 1 / math.sqrt(2)
        self.downsample = downsample

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale

class Discriminator(nn.Module):
    """
    Discriminator Network - ProGAN
    """
    def __init__(self, num_layers, channels_base=64, max_channels=256) -> None:
        super(Discriminator, self).__init__()
        features = [min(max_channels, channels_base * (2 ** i)) for i in range(num_layers - 1)]

        self.from_rgb = nn.Sequential(
            EQConv2d(3, channels_base, 1),
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = EQConv2d(final_features, final_features, 3)
        self.final = EQLinearLayer(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x):
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)
