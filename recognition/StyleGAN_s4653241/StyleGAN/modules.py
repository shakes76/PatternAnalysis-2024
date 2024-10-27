"""
Containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""
import torch
from torch.autograd.profiler_util import Kernel
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt


class MappingNetwork(nn.Module):
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
        x = x / torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + 1e-8)
        return self.mapping(x)

# Equalizer for the fully connected layer
'''
Linear layer with learning rate equalizing weights and bias
Returns the output of the linear transformation of the tensor with bias
'''
class EqualizerStraights(nn.Module):
    def __init__(self, in_chanel, out_chanel, bias=0):
        super().__init__()
        self.weight = EquilizerKG([out_chanel, in_chanel])
        self.bias = nn.Parameter(torch.ones(out_chanel) * bias)

    def forward(self, x):
        # Linear transformation
        return F.linear(x, self.weight(), bias = self.bias)


# Weight equalizer
'''
Maintains the weights in the network  at a similar scale during training.
Scale the weights at each layer with a constant such that,
 weight w' is scaled as w' = w * c where c is constant at each layer
'''
class EquilizerKG(nn.Module):
    def __init__(self,shape):
        super().__init__()
        # self.constanted =  1 / sqrt(torch.prod(shape[1:])) # yet to use
        self.constanted = 1 / sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.constanted

# -----------------Synthesis Network-----------------#

# Drip Block, basically Style Block
class DripBlock(nn.Module):
    def __init__(self,w_dim, in_chanel, out_chanel):
        super().__init__()

        self.need_face = EqualizerStraights(w_dim, in_chanel,bias=1)

        self.conv = Weight_Demod(in_chanel,out_chanel,kernel_size=3)

        # Noise and bias
        self.scale_noise  = nn.Parameter(torch.zeros(out_chanel))
        self.bias = nn.Parameter(torch.zeros(out_chanel))

        self.activation = nn.LeakyReLU(0.2)

    def forward(self,x,w,noise):
        s = self.need_face(w)
        x = self.conv(x,s)
        if noise is not None:
            x = x + self.scale_noise[None,:,None,None] * noise
        return self.activation(x + self.bias[None,:,None,None])


class Weight_Demod(nn.Module):
    def __init__(self, in_chanel, out_chanel, kernel_size, demodulate = True, eps = 1e-8):
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

    def __init__(self, W_DIM, features):
        super().__init__()
        self.to_style = EqualizerStraights(W_DIM, features, bias=1)

        self.conv = Weight_Demod(features, 3, kernel_size=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, w):

        style = self.to_style(w)
        x = self.conv(x, style)
        return self.activation(x + self.bias[None, :, None, None])
'''
Not implemented yet
'''
class Generator(nn.Module):
    def __init__(self, log_reso,w_dim, n_features = 32, max_features = 256):
        super().__init__()



        features = [min(max_features, n_features * (2 ** i)) for i in range(log_reso - 2, -1, -1)]
        self.n_blocks = len(features)

        # Initialize the trainable 4x4 constant tensor
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # First style block and it's rgb output. Initialises the generator.
        self.style_block = DripBlock(w_dim, features[0], features[0])
        self.to_rgb = ToRGB(w_dim, features[0])

        # Creates a series of Generator Blocks based on features length. 5 in this case.
        blocks = [GeneratorBlock(w_dim, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
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
            x = F.interpolate(x, scale_factor=2, mode="bilinear")
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            rgb = F.interpolate(rgb, scale_factor=2, mode="bilinear") + rgb_new

        # tanh is used to output rgb pixel values form -1 to 1
        return torch.tanh(rgb)


'''
Testing
'''
class GeneratorBlock(nn.Module):
    def __init__(self,w_dim, in_chanel, out_chanel):
        super().__init__()

        # First black changes the feature map size to the "out_chanel"
        self.styled_block1 = DripBlock(w_dim, in_chanel, out_chanel)
        self.styled_block2 = DripBlock(w_dim, out_chanel, out_chanel)

        self.to_rgb = ToRGB(w_dim, out_chanel)

    def forward(self, x,w,noise):

        # Style blocks with Noise Tensor
        x = self.styled_block1(x,w,noise[0])
        x = self.styled_block2(x,w,noise[1])

        # RBG img
        rgb = self.to_rgb(x,w)

        return x, rgb

#-----------Discriminator----------------#
'''
Implementation of the discriminator network
Is mostly the same as GAN discriminator network
'''
class Discriminator(nn.Module):
    def __init__(self, log_reso, n_features = 64, max_features = 256, rgb = True): # log resolution 2^log_reso= image size
        super().__init__()

        features = [min(max_features, n_features*(2**i)) for i in range(log_reso - 1)]

        # Not RGB unless trainined on CIFAR-10 images
        if rgb:
            self.rgb = nn.Sequential(
                Dis_Conv2d(3, n_features, 1),
                nn.LeakyReLU(0.2, True)
            )
        else:
            self.rgb = nn.Sequential(
                Dis_Conv2d(1, n_features, 1),
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
        self.final = EqualizerStraights(2 * 2 * final_features, 1)

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)


    def forward(self, x):
        x = self.rgb(x)
        x = self.blocks(x)

        x = self.minibatch_std(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.final(x)



class DiscriminatorBlock(nn.Module):
    def __init__(self,in_chanel,out_chanel,activation = nn.LeakyReLU(0.2,True),downsample = True):
        super().__init__()


        self.residual = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            Dis_Conv2d(in_chanel, out_chanel,kernel_size=1)
        )

        self.block = nn.Sequential(
            Dis_Conv2d(in_chanel, out_chanel, 3, padding = 1),
            activation,
            Dis_Conv2d(out_chanel, out_chanel, 3, padding = 1),
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
    def __init__(self,in_chanel,out_chanel,kernel_size,padding=0):
        super().__init__()
        self.padding = padding
        self.weight = EquilizerKG([out_chanel, in_chanel, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.zeros(out_chanel))

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias = self.bias, padding = self.padding)

#---------------------------#

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
