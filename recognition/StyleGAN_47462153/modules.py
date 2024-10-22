import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.ReLU(),
            nn.Linear(w_dim, w_dim)
        )
    
    def forward(self, z):
        return self.mapping(z)

class StyleGANGenerator(nn.Module):
    def __init__(self, z_dim, w_dim, img_channels):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, w_dim)
