"""
Containing the source code of the components of your model. Each component must be
implementated as a class or a function
"""
import torch
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

'''
Not implemented yet
'''
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

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
