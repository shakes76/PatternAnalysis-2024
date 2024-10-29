import torch
from torch.autograd.profiler_util import Kernel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from config import *

# Mapping Network used for mapping latent vector z to intermediate latent vector w
class MappingNetwork(nn.Module):
    """
    Initializes a mapping network to map a latent space to another latent space.
    
    Args:
        z_dim (int): Dimension of input latent vector z.
        w_dim (int): Dimension of intermediate latent vector w.
        activation (nn.Module): Activation function used in the mapping layers.
    """
    def __init__(self,z_dim,w_dim,activation = nn.ReLU()):
        
        super().__init__()

        # Mapping network
        self.mapping = nn.Sequential(
            EqualizerStraights(z_dim, w_dim),
            activation,
            EqualizerStraights(w_dim, w_dim),
            activation,
            EqualizerStraights(z_dim, w_dim),
            activation,
            EqualizerStraights(w_dim, w_dim),
            activation,
            EqualizerStraights(z_dim, w_dim),
            activation,
            EqualizerStraights(w_dim, w_dim),
            activation,
            EqualizerStraights(z_dim, w_dim),
            activation,
            EqualizerStraights(w_dim, w_dim)
        )

    def forward(self, x):
        # Normalize the input tensor
        x = x / torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + 1e-8) # for PixelNorm
        return self.mapping(x)

# Equalizer for the fully connected layer
class EqualizerStraights(nn.Module):
    """
    Fully connected layer with learnable weights and bias.
    
    Args:
        in_chanel (int): Input dimension.
        out_chanel (int): Output dimension.
        bias (float): Initial bias value.
    """
    def __init__(self, in_chanel, out_chanel, bias=0.0):
        
        super().__init__()
        self.weight = EquilizerKG([out_chanel, in_chanel])
        self.bias = nn.Parameter(torch.ones(out_chanel) * bias)

        self.input_size = in_chanel

    def forward(self, x):
        # Linear transformation
        weight = self.weight().to(x.device)
        return F.linear(x, weight, bias = self.bias.to(x.device))


# Weight equalizer
class EquilizerKG(nn.Module):
    """
    Weight equalizer to stabilize network weights across layers.
    
    Args:
        shape (list): Shape of the weight tensor.
    """
    def __init__(self,shape):
        
        super().__init__()
        # self.constanted =  1 / sqrt(torch.prod(shape[1:])) # yet to use
        self.constanted = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.constanted


# DripBlock for the generator, essentially a style block with noise and transformation
class DripBlock(nn.Module):
    """
    Style-based block with noise addition and activation.
    
    Args:
        W_DIM (int): Dimension of style vector w.
        in_chanel (int): Input channels.
        out_chanel (int): Output channels.
    """
    def __init__(self,W_DIM, in_chanel, out_chanel):
        
        super().__init__()

        self.need_face = EqualizerStraights(W_DIM, in_chanel,bias=1.0)

        self.conv = Weight_Demod(in_chanel,out_chanel,kernel_size=3)

        # Noise and bias
        self.scale_noise  = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_chanel))

        self.activation = nn.LeakyReLU(0.2,True)

    def forward(self,x,w,noise):
        s = self.need_face(w)
        x = self.conv(x,s)
        if noise is not None:
            x = x + self.scale_noise[None,:,None,None] * noise
        return self.activation(x + self.bias[None,:,None,None])


class Weight_Demod(nn.Module):
    """
    Weight demodulation for adaptive instance normalization.
    
    Args:
        in_chanel (int): Input channels.
        out_chanel (int): Output channels.
        kernel_size (int): Kernel size.
        demodulate (bool): Whether to apply weight demodulation.
        eps (float): Small value for numerical stability.
    """
    def __init__(
        self,
        in_chanel,
        out_chanel,
        kernel_size,
        demodulate = True,
        eps = 1e-8
    ):
    

        super().__init__()
        self.out_chanel = out_chanel
        self.demodulate = demodulate
        self.padding = (kernel_size -1) // 2

        # Weights with Equalizer learning rate
        self.weight = EquilizerKG([out_chanel, in_chanel, kernel_size, kernel_size])
        self.eps = eps # epsilon value for numerical stability

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
        weights = weights.reshape(b * self.out_chanel, *ws)

        # Group b is used to define a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # return x in shape of [batch_size, out_features, height, width]
        return x.reshape(-1, self.out_chanel, h, w)


class ToRGB(nn.Module):
    """
    Converts feature maps to RGB images in the generator.
    
    Args:
        W_DIM (int): Style vector dimension.
        features (int): Number of features.
    """
    def __init__(self, W_DIM, features):
        
        super().__init__()
        self.to_style = EqualizerStraights(W_DIM, features, bias=1.0)

        self.conv = Weight_Demod(features, CHANNELS, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(CHANNELS))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])


class Generator(nn.Module):
    """
    Progressive growing GAN generator.
    
    Args:
        log_resolution (int): Resolution level (e.g., log2 of image width).
        W_DIM (int): Style vector dimension.
        n_features (int): Base number of features.
        max_features (int): Maximum feature limit per layer.
    """
    def __init__(self, log_resolution, W_DIM, n_features = 32, max_features = 256):
        
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        self.style_block = DripBlock(W_DIM, features[0], features[0])
        self.to_rgb = ToRGB(W_DIM, features[0])

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


class GeneratorBlock(nn.Module):
    """
    A block within the generator, consisting of styled layers and RGB output conversion.

    Args:
        W_DIM (int): Dimensionality of the style vector `w`.
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
    """
    def __init__(self,W_DIM, in_chanel, out_chanel):
        super().__init__()

        # First black changes the feature map size to the "out_chanel"
        self.styled_block1 = DripBlock(W_DIM, in_chanel, out_chanel)
        self.styled_block2 = DripBlock(W_DIM, out_chanel, out_chanel)

        self.to_rgb = ToRGB(W_DIM, out_chanel)

    def forward(self, x,w,noise):

        # Style blocks with Noise Tensor
        x = self.styled_block1(x,w,noise[0])
        x = self.styled_block2(x,w,noise[1])

        # RBG img
        rgb = self.to_rgb(x,w)

        return x, rgb


# Discriminator for assessing real vs. generated images
class Discriminator(nn.Module):
    """
    The discriminator network for distinguishing real from generated images.

    Args:
        log_reso (int): Logarithmic resolution of the image (e.g., `log_reso=8` for 256x256 images).
        n_features (int): Initial number of features.
        max_features (int): Maximum number of features per layer.
    """
    def __init__(self, log_reso, n_features = 64, max_features = 256,): # log resolution 2^log_reso= image size
        super().__init__()

        features = [min(max_features, n_features*(2**i)) for i in range(log_reso - 1)]

        self.from_rgb = nn.Sequential(
            Dis_Conv2d(CHANNELS, n_features, 1), # Change here
            nn.LeakyReLU(0.2, True)
        )

        n_blocks = len(features) - 1

        # Squential blocks generated from size  for Discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i+1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        final_features = features[-1] + 1

        # Final conv layer with 3x3 kernel
        self.conv = Dis_Conv2d(final_features, final_features, 3)
        # Final Equalized linear layer for classification
        self.final = EqualizerStraights(2*2*final_features, 1)
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

class DiscriminatorBlock(nn.Module):
    """
    A block within the discriminator that includes convolutional layers, downsampling, and residual connections.

    Args:
        in_channel (int): Input channels.
        out_channel (int): Output channels.
        activation (nn.Module, optional): Activation function to use within the block. Defaults to LeakyReLU(0.2).
    """
    def __init__(self,in_chanel,out_chanel,activation = nn.LeakyReLU(0.2,True)):
        super().__init__()


        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Dis_Conv2d(in_chanel, out_chanel,kernel_size=1)
        )

        self.block = nn.Sequential(
            Dis_Conv2d(in_chanel, in_chanel, kernel_size=3, padding = 1),
            activation,
            Dis_Conv2d(in_chanel, out_chanel, kernel_size=3, padding = 1),
            activation
        )

        self.downsample = nn.AvgPool2d( kernel_size=2, stride=2)

        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.downsample(x)

        return (x + residual) * self.scale

#  Discriminator convolutional layer with equalized learning rate
class Dis_Conv2d(nn.Module):
    """
    Convolutional layer with equalized learning rate for the discriminator.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int): Kernel size.
        padding (int, optional): Padding for convolution. Defaults to 0.
    """
    def __init__(self,in_chanel,out_chanel,kernel_size,padding=0):
        super().__init__()
        # self.kernel_size = kernel_size
        self.padding = padding
        self.weight = EquilizerKG([out_chanel, in_chanel, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_chanel))

    def forward(self, x: torch.Tensor):

        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)

class PathLengthPenalty(nn.Module):
    """
    Path length penalty to regularize the generator by penalizing deviation in pixel norms.
    
    Args:
        beta (float): Decay rate for moving average of path length.
    """
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
