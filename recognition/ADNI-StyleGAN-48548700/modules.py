"""
=======================================================================
File Name: modules.py
Author: Baibhav Mund
Student ID: s48548700
Description:
    This file contains the implementation of key neural network components 
    used in a StyleGAN2 architecture. It includes the following classes:
    - MappingNetwork: A mapping network used to project latent space vectors
      (z) to the intermediate latent space (w).
    - Generator: A progressive generator that creates high-resolution images
      by progressively upscaling.
    - Discriminator: A multi-scale discriminator used for differentiating 
      between real and generated images.
    - Supporting classes for style modulation, convolution, and equalized 
      learning rate.

    These components are typically used in GAN architectures such as 
    StyleGAN or StyleGAN2 for generating high-quality images from noise vectors. 
    The generator progressively increases the resolution, while the 
    discriminator downscales the images for real/fake classification.

Usage:
    - Import the necessary components from this file for use in training GAN models.
    - Example usage:
        from modules import Generator, Discriminator, MappingNetwork
        gen = Generator(log_resolution=7, W_DIM=256)
        disc = Discriminator(log_resolution=7)

    - Use the MappingNetwork to map latent vectors from z-space to w-space 
      for style-based generation in the Generator.
    - The Generator and Discriminator classes are typically used with 
      noise inputs and progressively scaled images.

Classes:
    - MappingNetwork: Implements the latent space mapping to intermediate space.
    - Generator: Builds a progressive generator that outputs images.
    - GeneratorBlock: A basic block in the Generator for style-based image synthesis.
    - StyleBlock: Applies style-based modulation and noise injection to a feature map.
    - ToRGB: Converts a feature map to an RGB image at each resolution.
    - Conv2dWeightModulate: Implements weight modulation for style-based convolutions.
    - Discriminator: Classifies real and fake images by downscaling them progressively.
    - DiscriminatorBlock: A residual block in the Discriminator with downsampling.
    - EqualizedLinear: Linear layer with equalized learning rate.
    - EqualizedConv2d: Convolutional layer with equalized learning rate.
    - EqualizedWeight: Handles weight normalization in equalized learning rate layers.
    - PathLengthPenalty: Regularization class used for enforcing the path length penalty.

Dependencies:
    - PyTorch: Install via `pip install torch`
    - numpy: Install via `pip install numpy`
    
Notes:
    - Ensure that you adjust the log resolution and dimensions for both the 
      generator and discriminator based on your dataset's image size.
    - The discriminator uses a minibatch standard deviation for added variance.
    
=======================================================================
"""

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
import numpy as np

# Define a class for the Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        # Define a sequence of EqualizedLinear layers with ReLU activations
        self.mapping = nn.Sequential(
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim),
            nn.ReLU(),
            EqualizedLinear(z_dim, w_dim)
        )

    def forward(self, x):
        # PixelNorm: Normalize the input x
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return self.mapping(x)

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, log_resolution, W_DIM, n_features=32, max_features=256):
        super().__init__()
        # Determine the number of features for each block based on resolution
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        # Initialize the initial constant tensor as a learnable parameter
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # Create the StyleBlock for the initial layer and ToRGB for initial RGB output
        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

        # Create a list of GeneratorBlocks
        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
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

# Define the GeneratorBlock class
class GeneratorBlock(nn.Module):
    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()
        # Create two StyleBlocks and a ToRGB layer for each GeneratorBlock
        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)
        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        rgb = self.to_rgb(x, w)
        return x, rgb

# Define the StyleBlock class
class StyleBlock(nn.Module):
    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()
        # Define a linear layer for mapping the style vector
        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        # Define a convolutional layer with weight modulation and noise injection
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

# Defining the ToRGB class
class ToRGB(nn.Module):
    def __init__(self, W_DIM, features):
        super().__init__()
        # Define a linear layer for mapping the style vector
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)
        # Define a 1x1 convolutional layer with weight modulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):
        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])

# Defining the Conv2dWeightModulate class
class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, demodulate=True, eps=1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
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

# Defining the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, log_resolution, n_features=64, max_features=256):
        super().__init__()
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        # Create the initial from_rgb layers
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1
        # Create DiscriminatorBlocks for intermediate layers
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        final_features = features[-1] + 1
        # Create convolutional and linear layers for the final layer
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        # Calculate batch statistics (standard deviation) and append to the feature map
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

# Define the DiscriminatorBlock class
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Define a residual block with an AveragePool2d layer for downsampling
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), 
                                    EqualizedConv2d(in_features, out_features, kernel_size=1))
        # Define a main block with LeakyReLU activations
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )
        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # Downsample using AveragePool2d
        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)
        return (x + residual) * self.scale

# Define the EqualizedLinear class
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0.):
        super().__init__()
        # Define a linear layer with equalized learning rate
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

# Define the EqualizedConv2d class
class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding=0):
        super().__init__()
        # Define a convolutional layer with equalized learning rate
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

# Define the EqualizedWeight class
class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

# Define the PathLengthPenalty class
class PathLengthPenalty(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w, x):
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)

        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        return loss