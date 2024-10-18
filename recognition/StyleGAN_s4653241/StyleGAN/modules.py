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
    def __init__(self,z_dim,w_dim,activation = nn.ReLU):
        super().__init__()

        # Mapping network
        self.mapping = nn.Sequential(
            EqualizerStraights(z_dim, w_dim),
            activation(),
            EqualizerStraights(w_dim, w_dim),
            activation(),
            EqualizerStraights(z_dim, w_dim),
            activation(),
            EqualizerStraights(w_dim, w_dim),
            activation(),
            EqualizerStraights(z_dim, w_dim),
            activation(),
            EqualizerStraights(w_dim, w_dim),
            activation(),
            EqualizerStraights(z_dim, w_dim),
            activation(),
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

'''
Not implemented yet
'''
class SynthesisNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

'''
Not implemented yet
'''
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

#-----------Discriminator----------------#
'''
Implementation of the discriminator network
Is mostly the same as GAN discriminator network
'''
class Discriminator(nn.Module):
    def __init__(self, log_reso, n_features = 64, max_features = 256, rgb = False): # log resolution 2^log_reso= image size
        super().__init__()

        features = [min(max_features, n_features*(2**i)) for i in range(log_reso - 1)]

        # Not RGB unless trainined on CIFAR-10 images
        if rgb:
            self.rgb = nn.Sequential(
                Dis_Conv2d(3, n_features, 1),
                nn.LeakyReLU(0.2, True)
            )
        else:
            self.grey = nn.Sequential(
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
            activation(),
            Dis_Conv2d(out_chanel, out_chanel, 3, padding = 1),
            activation()
        )

        self.downsample = nn.AvgPool2d( kernel_size=2, stride=2)

        self.scale = 1 / sqrt(2)

    def forward(self, x):
        residual = self.residual(x)

        x = self.block(x)
        x = self.down_sample(x)

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



'''
Not implemented yet
'''
class StyleGAN2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

'''
Not implemented yet
'''
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass
