import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(z_dim , w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

        def forward(self, z):
            return self.mapping(z)


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def foward(self, image, noise):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


