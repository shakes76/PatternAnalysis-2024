"""
Source code of model components.
Created on 08/10/2024
Reference: https://blog.paperspace.com/implementation-stylegan2-from-scratch/
A Style-Based Generator Architecture for Generative Adversarial Networks https://arxiv.org/pdf/1812.04948
Analyzing and Improving the Image Quality of StyleGAN https://arxiv.org/pdf/1912.04958
StyleGan2 official pytorch implementation https://github.com/NVlabs/stylegan2-ada-pytorch
PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION https://arxiv.org/pdf/1710.10196
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from math import log2, sqrt
import numpy as np

"""
"""
class MappingNetwork(nn.Module):
    """
    Mapping network that outputs latent vector to be the inputs of Generator
    """
    def __init__(self, z_dim, w_dim):
        super().__init__()
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
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm
        return self.mapping(x)


class Generator(nn.Module):
    """
    Generator of The StyleGan Network
    Consists of a learned constant, a StyleBlock and a ToRGB block, and multiple generator blocks.
    """

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):
        """
        :param log_resolution: log2 of the output image resolution
        :param W_DIM: The dimension of the style vector
        :param n_features: number of features in the highest convolution layer
        :param max_features: maximum amount of features in any convolution layer
        """
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):
        """
        Takes in a style vector and input noise, do convolution through the style block and every generator block, and
        returns a RGB image.
        :param w: style vector, which has the shape [n_blocks, Batch size, W_DIM]
        :param input_noise: input noise for each block
        :return: The last RGB image last highest layer, with tanh as the activation function
        """

        batch_size = w.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        return torch.tanh(rgb)

class GeneratorBlock(nn.Module):
    """
    A generator block consists of two style block and a ToRGB block to have RGB output
    """
    def __init__(self, W_DIM, in_features, out_features):
        """
        :param W_DIM: The dimension of style vector
        :param in_features: number of incoming features
        :param out_features: number of resultant features
        """

        super().__init__()

        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):
        """
        :param x: the input feature map
        :param w: style vector with shape [batch_size, W_DIM]
        :param noise: a tuple of two noise tensor
        :return:
        """

        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, w)

        return x, rgb


class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):

        super().__init__()

        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        """
        Runs the input feature map and style vector through the Weighted Modulation.
        :param x: input feature map
        :param w: style
        :param noise: noise tensor of size [batch_size, 1, height, width]
        :return: feature map
        """

        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    """
    Transform the input into a RGB image
    """

    def __init__(self, W_DIM, features):

        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)

        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])

class EqualizedWeight(nn.Module):
    """
    Equalized learning rate, wˆ_i = w_i/c, where w_i are the weights and c is per-layer normalisation constant from
    He's initializer.
    """

    def __init__(self, shape):

        super().__init__()

        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

class EqualizedLinear(nn.Module):
    """
    Equalizes the learning rate for a linear layer
    """

    def __init__(self, in_features, out_features, bias = 0.):

        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedConv2d(nn.Module):
    """
    Equalizes the learning rate for a convolution layer
    """

    def __init__(self, in_features, out_features,
                 kernel_size, padding = 0):

        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class Conv2dWeightModulate(nn.Module):
    """
    Scales the convolution weights by the style vector and demodulates it.
    Without demodulation, the style becomes cumulative rather layer specific.
    """
    def __init__(self, in_features, out_features, kernel_size, demodulate = True, epsilon = 1e-8):
        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size-1)//2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.epsilon = epsilon

    def forward(self, x, s):

        b, _, h, w = x.shape

        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.epsilon)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        return x.reshape(-1, self.out_features, h, w)


class Discriminator(nn.Module):
    """
    The discriminator of the StyleGAN2 model.
    Essentially, it is a classification network that outputs whether an input image is real or fake.
    """

    def __init__(self, log_resolution, n_features = 64, max_features = 256):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

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
        data = x
        x = x.reshape(x.shape[0], -1)
        return self.final(x)


class DiscriminatorBlock(nn.Module):
    """
    Each discriminator block consists of 2 3x3 convolution
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = nn.AvgPool2d(
            kernel_size=2, stride=2
        )  # down sampling using avg pool

        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


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


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def get_w(batch_size, w_dim, device, mapping_network, log_resolution):
    """
    Sample a z randomly and gets w from the mapping network
    """
    z = torch.randn(batch_size, w_dim).to(device)
    w = mapping_network(z)
    return w[None, :, :].expand(log_resolution, -1, -1)

def get_noise(batch_size, log_resolution, device):
    """
    Generates noise for each generator block
    :returns the generated noise
    """

    noise = []
    resolution = 4

    for i in range(log_resolution):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        noise.append((n1, n2))

        resolution *= 2

    return noise
