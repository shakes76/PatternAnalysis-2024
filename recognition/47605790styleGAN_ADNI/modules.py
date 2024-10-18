"""
This file contains all the main components of StyleGAN including:
- Mapping Network
- Adaptive Instance Normalization Layer (AdaIN)
- Noise Injection Layer
- StyleGAN Generator and Discriminator
"""

import torch
import torch.nn as nn

# MappingNetwork: Converts a latent vector (z) into an intermediate space (w) through multiple fully connected layers.
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, n_layers=8):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(latent_dim, latent_dim) for _ in range(n_layers)]
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Forward pass for the mapping network.
        Args:
            z: Latent vector.
        Returns:
            w: Mapped latent vector in intermediate space.
        """
        return self.mapping(z)

# AdaIN: Adaptive Instance Normalization, modulates feature maps using style information from w.
class AdaIN(nn.Module):
    def __init__(self, style_dim, channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.fc = nn.Linear(style_dim, channels * 2)  # Learnable scale and bias
    
    def forward(self, x, style):
        """
        Applies Adaptive Instance Normalization to the input feature maps.
        Args:
            x: Input feature map.
            style: Style vector for modulation.
        Returns:
            Modulated feature maps.
        """
        normalized = self.norm(x)
        style = self.fc(style).view(-1, 2, x.size(1), 1, 1)
        scale, bias = style[:, 0], style[:, 1]
        return scale * normalized + bias

# NoiseInjection: Adds random noise to the input for increased variation in the generated images.
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x, noise=None):
        """
        Injects noise into the input to introduce variation.
        Args:
            x: Input feature map.
            noise: Optional precomputed noise; if not provided, generated internally.
        Returns:
            Noisy input.
        """
        if noise is None:
            noise = torch.randn(x.size(), device=x.device)
        return x + self.weight * noise

# ModulatedConv2d: Modulated convolution with style-based scaling, controlling the convolution weights.
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super(ModulatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.style = nn.Linear(style_dim, in_channels)  # Style modulation
        self.demodulate = demodulate
    
    def forward(self, x, style):
        """
        Applies modulated convolution to the input.
        Args:
            x: Input feature map.
            style: Style vector for scaling the convolution filters.
        Returns:
            Output feature map after convolution.
        """
        mod = self.style(style).view(-1, x.size(1), 1, 1)
        x = x * mod
        out = self.conv(x)
        
        if self.demodulate:
            out = out / torch.sqrt(torch.mean(out ** 2, dim=[1, 2, 3], keepdim=True) + 1e-8)
        
        return out

# Generator: The core model that generates images using modulated convolutions, noise injection, and AdaIN.
class Generator(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, img_size=128, channels=3):
        super(Generator, self).__init__()
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ModulatedConv2d(128, 128, 3, style_dim),
                NoiseInjection(128),
                AdaIN(style_dim, 128),
                nn.Upsample(scale_factor=2)
            ),
            nn.Sequential(
                ModulatedConv2d(128, 64, 3, style_dim),
                NoiseInjection(64),
                AdaIN(style_dim, 64),
                nn.Upsample(scale_factor=2)
            ),
            nn.Sequential(
                ModulatedConv2d(64, 32, 3, style_dim),
                NoiseInjection(32),
                AdaIN(style_dim, 32),
                nn.Conv2d(32, channels, 3, padding=1),
                nn.Tanh()
            )
        ])
    
    def forward(self, z, style):
        """
        Forward pass for the generator.
        Args:
            z: Latent vector.
            style: Style vector for modulating feature maps.
        Returns:
            Generated image.
        """
        out = self.l1(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        
        for block in self.blocks:
            out = block[0](out, style)  # ModulatedConv2d needs style as an argument
            out = block[1](out)  # NoiseInjection
            out = block[2](out, style)  # AdaIN needs style input
            if len(block) > 3:
                out = block[3](out)  # Upsample or other layers
        return out

# Discriminator: Classifies whether an image is real or generated.
class Discriminator(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Output layer
        self.adv_layer = nn.Sequential(nn.Linear(128 * (img_size // 16) ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        """
        Forward pass for the discriminator.
        Args:
            img: Input image.
        Returns:
            Validity score (real or fake).
        """
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Loss function for adversarial training
adversarial_loss = nn.BCELoss()
