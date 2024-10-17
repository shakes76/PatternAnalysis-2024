'''
This file contain all the main component of StyleGAN including Mapping Network, 
Adaptive Instance Normalization Layer (AdaIN), 
Noise Injection Layer with the StyleGAN Generator and Discriminator 
'''

import torch
import torch.nn as nn

# Mapping Network for transforming latent vector z to intermediate space w
class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, n_layers=8):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(latent_dim, latent_dim) for _ in range(n_layers)]
        self.mapping = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.mapping(z)

# Adaptive Instance Normalization (AdaIN) Layer
class AdaIN(nn.Module):
    def __init__(self, style_dim, channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.fc = nn.Linear(style_dim, channels * 2)  # Learnable scale and bias
    
    def forward(self, x, style):
        normalized = self.norm(x)
        style = self.fc(style).view(-1, 2, x.size(1), 1, 1)
        scale, bias = style[:, 0], style[:, 1]
        return scale * normalized + bias

# Noise Injection Layer
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(), device=x.device)
        return x + self.weight * noise

# StyleGAN Generator with Progressive Growing and Modulated Convolutions
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True):
        super(ModulatedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.style = nn.Linear(style_dim, in_channels)  # Style modulation
        self.demodulate = demodulate
    
    def forward(self, x, style):
        # Style modulation
        mod = self.style(style).view(-1, x.size(1), 1, 1)
        x = x * mod
        
        # Convolution
        out = self.conv(x)
        
        if self.demodulate:
            out = out / torch.sqrt(torch.mean(out ** 2, dim=[1, 2, 3], keepdim=True) + 1e-8)
        
        return out

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
        out = self.l1(z)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        
        for block in self.blocks:
            out = block[0](out, style)  # ModulatedConv2d needs style as an argument
            out = block[1](out)  # NoiseInjection
            out = block[2](out, style)  # AdaIN needs style input
            if len(block) > 3:
                out = block[3](out)  # Upsample or other layers
        return out

# Discriminator model
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
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

# Loss functions
adversarial_loss = nn.BCELoss()
