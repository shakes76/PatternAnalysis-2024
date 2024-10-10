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

class NoiseInjection(nn.Module):
    def __init__(self):
        """
        Injects random noise into the feature map to generate fine details
        """
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))  # Learnable noise weight

    def forward(self, x, noise=None):
        """
        Adds noise to the input feature map
        Args:
            x: Input feature map
            noise: Random noise tensor to be added
        """
        if noise is None:
            batch, _, height, width = x.size()
            noise = torch.randn(batch, 1, height, width).to(x.device) 
        return x + self.weight * noise
    
class SynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim, upsample=True):
        """
        Synthesis Block with convolutional layers and residual connections
        Args:
            in_channels: Number of input channels from the previous layer
            out_channels: Number of output channels
            style_dim: Dimension of style vector w
            upsample: Boolean to indicate if upsampling is needed
        """
        super(SynthesisBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2) if upsample else None
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.noise3 = NoiseInjection() 
        
        # AdaIN and style control for each convolution layer
        self.style1 = nn.Linear(style_dim, out_channels)
        self.style2 = nn.Linear(style_dim, out_channels)
        self.style3 = nn.Linear(style_dim, out_channels)  
        
        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, w):
        """
        Forward pass through the enhanced synthesis block
        Args:
            x: Input feature map
            w: Style vector from the mapping network
        Returns:
            Upsampled and style-modulated feature map
        """
        if self.upsample:
            x = self.upsample(x)  # Upsample the feature map to a higher resolution
        
        # First convolution layer
        residual = self.residual(x)  # Apply residual connection
        x = F.relu(self.conv1(x))
        x = self.noise1(x)  # Add noise to first conv output
        x = adain(x, self.style1(w))  # Apply AdaIN to the first conv layer
        
        # Second convolution layer
        x = F.relu(self.conv2(x))
        x = self.noise2(x)  # Add noise to second conv output
        x = adain(x, self.style2(w))  # Apply AdaIN to the second conv layer

        # Third convolution layer
        x = F.relu(self.conv3(x))
        x = self.noise3(x)  # Add noise to third conv output
        x = adain(x, self.style3(w))  # Apply AdaIN to the third conv layer

        # Add residual connection
        return x + residual  # Add residual connection for better gradient flow
