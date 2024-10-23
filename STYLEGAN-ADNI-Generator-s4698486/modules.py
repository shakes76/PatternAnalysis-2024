#modules.py

import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt #math sqrt is about 7 times faster than numpy sqrt
import numpy as np

from constants import channels, interpolation


"""
 Mapping network with 8 Linear layers
 Just maps the latent space to style space, as outlined in StyleGAN ideology.
"""
class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()

        # Create a Sequential container with 8 Equalized Linear layer using ReLU activ fn
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
        # Normalize z
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)  # for PixelNorm 
        # Maps z to w
        return self.mapping(x)


"""
 Linear layer - just with equal weight and bias. Empirically works better with MappingNetwork
"""
class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)

# Weight equalization layer
"""
 Normalises weights to ensure that they are at a similar scale throughout training.
 Minimises potential for NaN losses.
"""
class EqualizedWeight(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.c = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c



"""
 The generator starts with an initial learned constant, which is a StyleGAN2 quirk
 We then combine GeneratorBlocks to upscale the features of the image, before normalising it to [-1, 1]
 output using tanh.
"""
class Generator(nn.Module):

    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):
        super().__init__()

        # Define a series of progressively increasing features from n_features to max_features for each block.
        # [32, 64, 128, 256, 256]
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        # Initialize the trainable 4x4 constant tensor
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # First style block and it's rgb output. Initialises the generator.
        self.style_block = StyleBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

        # Creates a series of Generator Blocks based on features length. 5 in this case.
        blocks = [GeneratorBlock(W_DIM, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, w, input_noise):
        batch_size = w.shape[1]

        # Expand the learnt constant to match the batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # Get the first style block and the rgb img
        x = self.style_block(x, w[0], input_noise[0][1])
        rgb = self.to_rgb(x, w[0])

        # Rest of the blocks upsample the img using interpolation set in the config file and add to the rgb from the block
        for i in range(1, self.n_blocks):
            x = F.interpolate(x, scale_factor=2, mode=interpolation)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode=interpolation) + rgb_new

        # tanh is used to output rgb pixel values form -1 to 1
        return torch.tanh(rgb)

'''
The generator block consists of two style blocks and a 3x3 convolutions with style modulation
Returns the feature map and an RGB img.
'''
class GeneratorBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()

        # First block changes the feature map size to the 'out features'
        self.style_block1 = StyleBlock(W_DIM, in_features, out_features)
        self.style_block2 = StyleBlock(W_DIM, out_features, out_features)

        self.to_rgb = ToRGB(W_DIM, out_features)

    def forward(self, x, w, noise):

        # Style blocks with Noise tensor
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])
        
        # get RGB img
        rgb = self.to_rgb(x, w)

        return x, rgb
    
# Style block has a weight modulation convolution layer.
class StyleBlock(nn.Module):

    def __init__(self, W_DIM, in_features, out_features):
        super().__init__()
        
        # Get style vector from equalized linear layer
        self.to_style = EqualizedLinear(W_DIM, in_features, bias=1.0)
        # Weight Modulated conv layer 
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # Noise and bias
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w, noise):
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])

'''
Generates an RGB image from a feature map using 1x1 convolution.
Uses the style vector from the mapping network through the Equalized Linear layer
'''    
class ToRGB(nn.Module):

    def __init__(self, W_DIM, features):

        super().__init__()
        self.to_style = EqualizedLinear(W_DIM, features, bias=1.0)
        # Weight modulated conv layer without demodulation
        self.conv = Conv2dWeightModulate(features, channels, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])

'''
This layer scales the convolution weights by the style vector and demodulates by normalizing it.    
'''
class Conv2dWeightModulate(nn.Module):

    def __init__(self, in_features, out_features, kernel_size,
                 demodulate = True, eps = 1e-8):

        super().__init__()
        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        # Weights with Equalized learning rate
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.eps = eps  #epsilon

    def forward(self, x, s):

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        # The result has shape [batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Weight Demodulation
        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        # Reshape x and weights
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Group b is used to define a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # return x in shape of [batch_size, out_features, height, width]
        return x.reshape(-1, self.out_features, h, w)
    
"""
 Discriminator first transforms the image to a feature map of the same resolution and then
 runs it through a series of blocks with residual connections.
 The resolution is down-sampled by 2x at each block while doubling the number of features.
"""
class Discriminator(nn.Module):

    def __init__(self, log_resolution, n_features = 64, max_features = 256):
        super().__init__()

        # Calculate the number of features for each block.
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        # Layer to convert RGB image to a feature map with `n_features`.
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(channels, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )
        n_blocks = len(features) - 1

        # A sequential container for Discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1
        # Final conv layer with 3x3 kernel
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        # Final Equalized linear layer for classification
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    """
     Mini-batch standard deviation calculates the standard deviation
     across a mini-batch for each feature in the feature map. This is used to prevent
     against the generator generating images with very minimal standard deviation (the same images)
    """
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
    
# Discriminator block consists of two $3 \times 3$ convolutions with a residual connection.
class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        # Down-sampling with AvgPool with 2x2 kernel and 1x1 convolution layer for the residual connection
        self.residual = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2), # down sampling using avg pool
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        # 2 conv layer with 3x3 kernel
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # down sampling using avg pool
        self.down_sample = nn.AvgPool2d( kernel_size=2, stride=2)  

        # Scaling factor after adding the residual
        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


"""
 Adds weight equalisation to conv2d block, in order to enhance stabilisation of training
"""
class EqualizedConv2d(nn.Module):

    def __init__(self, in_features, out_features, kernel_size, padding = 0):
        super().__init__()
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

"""
 Path length penalty regularisation, which ensures that a change in the latent vector corresponds to a proportional
 change to the generated image. The intention of this is to disentangle features in the style space further.
"""
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

        # Scaling
        output = (x * y).sum() / sqrt(image_size)
        sqrt(image_size)

        # Computes gradient
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        # Calculated L2-norm
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regulatrise after first step
        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummpy loss tensor if computation fails
            loss = norm.new_tensor(0)

        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)

        # return the penalty
        return loss