import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        # Normalizing input by dividing by its mean square root to stabilize training
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        # instance normalization followed by learned style scale and bias transformations
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias

class WSLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Weight scaling factor for normalization
        self.scale = (2 / in_features) ** 0.5  
        self.bias = self.linear.bias
        self.linear.bias = None
        # Initialize weights with normal distribution
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        # Initialize bias to zero  
        nn.init.zeros_(self.bias)  

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5  
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight, mean=0.0, std=1.0)  
        nn.init.zeros_(self.bias)  
    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1)

class InjectNoise(nn.Module):
    def __init__(self, channels):
        super(InjectNoise, self).__init__()
        # Learnable parameter to scale noise
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))  

    def forward(self, x):
        # Adding scaled random noise to the input feature map
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        return x + self.weight * noise

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        # Passing input through two convolution layers with AdaIN and noise injection
        x = self.conv1(x)
        x = self.inject_noise1(x)
        x = self.leaky(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.inject_noise2(x)
        x = self.leaky(x)
        x = self.adain2(x, w)
        return x

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super(MappingNetwork, self).__init__()
        layers = [PixelNorm()]
        # 8 fully connected layers as per StyleGAN
        for _ in range(8):
            layers.append(WSLinear(z_dim if _ == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        # Mapping latent vector z to intermediate latent vector w
        return self.mapping(x)

class Generator(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, in_channels=512, img_channels=1):
        super(Generator, self).__init__()
        # Learnable constant input
        self.initial_constant = nn.Parameter(torch.ones(1, in_channels, 4, 4))  
        self.initial_noise = InjectNoise(in_channels)
        self.initial_conv = WSConv2d(in_channels, in_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.initial_adain = AdaIN(in_channels, w_dim)

        self.progression_blocks = nn.ModuleList([
            GenBlock(in_channels, in_channels, w_dim),  # 8x8
            GenBlock(in_channels, in_channels, w_dim),  # 16x16
            GenBlock(in_channels, in_channels, w_dim),  # 32x32
            GenBlock(in_channels, in_channels, w_dim),  # 64x64
        ])

        # Convert features to RGB image
        self.to_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, padding=0)  

    def forward(self, noise):
        # Forward pass through mapping and synthesis networks
        w = self.mapping(noise)
        # Expand constant to batch size
        x = self.initial_constant.repeat(noise.size(0), 1, 1, 1)  
        x = self.initial_noise(x)
        x = self.leaky(self.initial_conv(x))
        x = self.initial_adain(x, w)

        for block in self.progression_blocks:
            # Upsample feature map
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')  
            x = block(x, w)  

        # Generate final image
        out = self.to_rgb(x)  
        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, in_channels)
        self.conv2 = WSConv2d(in_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Pass input through two convolutional layers with LeakyReLU activations
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=512, img_channels=1):
        super(Discriminator, self).__init__()
        self.from_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, padding=0)  
        self.progression_blocks = nn.ModuleList([
            ConvBlock(in_channels, in_channels),  
            ConvBlock(in_channels, in_channels),  
            ConvBlock(in_channels, in_channels),  
            ConvBlock(in_channels, in_channels),  
        ])

        self.leaky = nn.LeakyReLU(0.2)
        self.final_conv = WSConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # Final linear layer for real/fake prediction
        self.linear = WSLinear(in_channels * 4 * 4, 1)  

    def forward(self, x):
        # Forward pass through the discriminator network
        x = self.from_rgb(x)  
        for block in self.progression_blocks:
            x = block(x) 
            # Downsample feature map 
            x = nn.functional.avg_pool2d(x, 2) 

        # Final convolution with activation 
        x = self.leaky(self.final_conv(x))  
        x = x.view(x.size(0), -1)  
        # Predict real or fake
        x = self.linear(x)
        return x
