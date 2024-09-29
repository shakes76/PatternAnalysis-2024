"""
A implementation of a Style Generative Adversarial Network 2 (StyleGAN2) designed for 256x256 greyscale images.
References:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://github.com/indiradutta/DC_GAN
https://arxiv.org/abs/1511.06434
https://github.com/NVlabs/stylegan3
https://arxiv.org/pdf/1812.04948 
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
import numpy as np
import torch.nn.functional as F
import math


class FullyConnectedLayer(nn.Module):
    """
    A flexible fully connected layer with various customization options.
    This layer can be used in the mapping network and other parts of StyleGAN2.
    """
    def __init__(self, 
                 in_features,               # Num input features.
                 out_features,              # Num output features.
                 bias=True,                 # Include bias term?
                 activation='linear',       # Activation function to use.
                 weight_init='xavier',      # Weight initialisation method.
                 dropout=0.0,               # Dropout rate.
                 batch_norm=False,          # Use batch normalisation?
                 layer_norm=False           # Use layer normalisation?
                ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        
        # Initialise weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters(weight_init)
        self.act_fn = self.get_activation_fn(activation)

    def reset_parameters(self, weight_init):
        """Initialise the weights using the specified method."""
        if weight_init == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        elif weight_init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        elif weight_init == 'orthogonal':
            nn.init.orthogonal_(self.weight)
        else:
            raise ValueError(f"Unsupported weight initialisation: {weight_init}")
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_activation_fn(self, activation):
        """Return the specified activation function."""
        if activation == 'relu':
            return F.relu
        elif activation == 'leaky_relu':
            return lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif activation == 'elu':
            return F.elu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'swish':
            return lambda x: x * torch.sigmoid(x)
        elif activation == 'linear':
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        """Forward pass of the fully connected layer."""
        x = F.linear(x, self.weight, self.bias)
        
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        
        x = self.act_fn(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        return x

