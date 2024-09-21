import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import VAEResidualBlock, VAEAttentionBlock

class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        pass



class VAE_Encoder(nn.Sequential):
    """
    Variational Auto-Encoder Encoder to compress images to latent space

    Args:
        in_channels (int): number of channels in the input image
        latent_dim (int): dimension of the latent space

    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Output: [B, latent_dim, 4, 4] tensor of latent space representation
    """

    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__(

            # Downsample from (H, W) to (H/8, W/8)
            nn.Conv2d(in_channels, 128, 3, padding=1), # (B, C, H, W) -> (B, 128, H, W)
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),

            nn.Conv2d(128, 128, 3, stride=2, padding=0), # (B, 128, H, W) -> (B, 256, H/2, W/2)
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),

            nn.Conv2d(256, 256, 3, stride=2, padding=0), # (B, 256, H/2, W/2) -> (B, 512, H/4, W/4)
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),

            nn.Conv2d(512, 512, 3, stride=2, padding=0), # (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),

            nn.GroupNorm(32, 512), 
            nn.SiLU(),

            # Restrict to latent dimension
            nn.Conv2d(512, latent_dim, 3, stride=1, padding=1), # (B, 512, H/8, W/8) -> (B, latent_dim, H/8, W/8)
            nn.Conv2d(latent_dim, latent_dim, 3, stride=1, padding=0), 
        )

    def forward(self, x, noise):
        """
        Encode image to latent space with given noise

        Args:
            x (torch.Tensor): [B, C, H, W] tensor of images
            noise (torch.Tensor): [B, latent_dim, H/8, W/8] tensor of noise

        Returns:
            torch.Tensor: [B, latent_dim, 4, 4] tensor of latent space representation
        """
        for layer in self:
            if getattr(layer, 'stride', None) == 2:
                 x = F.pad(x, (0, 1, 0, 1)) # just add padding to H and W
            x = layer(x, noise)

        mean, log_var = torch.chunk(x, 2, dim=1) 
        variance = torch.clamp(log_var, -30, 20).exp()
        std = variance.sqrt()

        # Sample from N(mean, std) of the latent distribution
        z = mean + std * noise
        z *= 0.18215 # multiple by constant from original paper

        return z
    


    

    
