"""
@brief: This file contains the implementation of the main components for training a StyleGAN, 
including the generator, discriminator, mapping network, AdaIN normalization, and noise injection. 
It also includes functions to save model checkpoints.

@Author: Amlan Nag (s4721964)
"""
import torch
import torch.nn.functional as F

from torchvision.utils import save_image
from params import *
from torch import nn


#Compressing the chanels from 4,8,16,32,128,256
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8]

class PixelNorm(nn.Module):
    """
    Normalizes each pixel of the input tensor using its mean across channels to 
    stabalise the gradient
    """
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class WSLinear(nn.Module):
    """
    Weighted scaled linear layer with initialized weights for the StyleGAN generator 
    and discriminator
    """
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class MappingNetwork(nn.Module):
    """
    Maps the random noise vector (Z) to the latent space (W) by passing it through 
    several fully connected layers. This turns the noise into a style code.
    """
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixelNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )


    def forward(self, x):
        return self.mapping(x)


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer normalizes activations and adjusts 
    them based on learned style parameters.
    """
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


class InjectNoise(nn.Module):
    """
    Injects random noise into the input, helping to create variations in image generation.
    """
    def __init__(self, channels):
        super(InjectNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight * noise


class WSConv2d(nn.Module):
    """
    Weighted scaled 2D convolutional layer which stabilizes training
    for StyleGAN by scaling weights properly
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(nn.Module):
    """
    Convolutional block with two WSConv2d layers and LeakyReLU activation. 
    Used in both generator and discriminator
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x


class Discriminator(nn.Module):
    """
    StyleGAN Discriminator responsible for determining whether the generated images are real or fake. 
    It downsamples the image progressively as it passes through different layers.
    """
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])
            conv_out = int(in_channels * factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in, kernel_size=1))

        self.initial_rgb = WSConv2d(img_channels, in_channels, kernel_size=1)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels, kernel_size=3),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1),
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


class GenBlock(nn.Module):
    """
    Generator block with two WSConv2d layers, noise injection, and AdaIN layers.
    Each block takes a latent vector and transforms it progressively to a final image.
    """
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
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x


class Generator(nn.Module):
    """
    StyleGAN Generator progressively increases the size of generated images using 
    multiple blocks, starting from a 4x4 image to the 256x256.
    """
    def __init__(self, z_dim, w_dim, in_channels, img_channels=1):
        super(Generator, self).__init__()
        self.starting_constant = nn.Parameter(torch.ones((1, in_channels, 4, 4)))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain1 = AdaIN(in_channels, w_dim)
        self.initial_adain2 = AdaIN(in_channels, w_dim)
        self.initial_noise1 = InjectNoise(in_channels)
        self.initial_noise2 = InjectNoise(in_channels)
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1)
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i + 1])
            self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim))
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.starting_constant), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        if steps == 0:
            return self.initial_rgb(x)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="bilinear")
            out = self.prog_blocks[step](upscaled, w)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)
        return self.fade_in(alpha, final_upscaled, final_out)


def save_model(gen, disc, opt_gen, opt_disc, epoch, step, disc_losses, gen_losses, file_path="model_checkpoint.pth"):
    """
    Saves the model's generator, discriminator, optimizer states, losses, and training steps to a file.
    """
    checkpoint = {
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "opt_gen_state_dict": opt_gen.state_dict(),
        "opt_disc_state_dict": opt_disc.state_dict(),
        "epoch": epoch,
        "step": step,
        "disc_losses": disc_losses,
        "gen_losses": gen_losses
    }
    torch.save(checkpoint, file_path)
    print(f"Model and losses saved to {file_path}")

