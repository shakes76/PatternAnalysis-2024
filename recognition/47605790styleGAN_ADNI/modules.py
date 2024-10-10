import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=12):
        """
        The Mapping Network converts a latent vector z into a disentangled latent space w (style)
        Args:   z_dim: Dimension of input latent vector z
                w_dim: Dimension of output latent vector w
                num_layers: Number of fully connected layers in the mapping network
            
        """
        super(MappingNetwork, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(z_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))  # LeakyReLU activation for better gradient flow
            layers.append(nn.BatchNorm1d(w_dim))  # Batch normalization for stable training
            layers.append(nn.Dropout(0.3))  # Dropout to prevent overfitting
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        """
        Forward pass through mapping network
        Args:   z: Latent vector z from normal distribution (N(0, 1))
            
        Returns:
                w: Disentangled latent vector w
        """
        return self.mapping(z)

def adain(feature_map, style):
    """
    Adaptive Instance Normalization (AdaIN)
    Args:   feature_map: Input feature map to be normalized
            style: Style vector w used to scale and shift the feature map
        
    Returns:
            Normalized feature map
    """
    size = feature_map.size()
    mean, std = style[:, :size[1]], style[:, size[1]:]
    mean = mean.unsqueeze(2).unsqueeze(3)  # Adding spatial dimensions to mean
    std = std.unsqueeze(2).unsqueeze(3)  # Adding spatial dimensions to std
    
    # Normalize the input feature map
    normalized = (feature_map - feature_map.mean([2, 3], keepdim=True)) / (feature_map.std([2, 3], keepdim=True) + 1e-8)
    
    # Apply style modulation
    return std * normalized + mean
