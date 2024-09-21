import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import VAEResidualBlock, VAEAttentionBlock

class VAE(nn.Module):

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAE_Encoder(in_channels, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, in_channels)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar
    
    def sample(self, mean, logvar):
        """
        Sample a new image from the latent space defined by mean, logvar
        """
        std = torch.exp(0.5 * logvar)
        noise1 = torch.randn_like(mean)
        noise2 = torch.randn_like(mean)
        z1 = mean + std * noise1
        z2 = mean + std * noise2
        z = torch.cat([z1, z2], dim=1)  
        z *= 0.18215
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        return self.decode(z), mu, logvar


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

            nn.Conv2d(128, 128, 3, stride=2, padding=1), # (B, 128, H, W) -> (B, 256, H/2, W/2)
            VAEResidualBlock(128, 256),
            VAEResidualBlock(256, 256),

            nn.Conv2d(256, 256, 3, stride=2, padding=1), # (B, 256, H/2, W/2) -> (B, 512, H/4, W/4)
            VAEResidualBlock(256, 512),
            VAEResidualBlock(512, 512),

            nn.Conv2d(512, 512, 3, stride=2, padding=1), # (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),

            nn.GroupNorm(32, 512), 
            nn.SiLU(),

            # Restrict to latent dimension
            nn.Conv2d(512, latent_dim, 3, stride=1, padding=1), # (B, 512, H/8, W/8) -> (B, latent_dim, H/8, W/8)
            nn.Conv2d(latent_dim, latent_dim, 3, stride=1, padding=1), 
        )

    def forward(self, x):
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
            x = layer(x)
        mean, log_var = torch.chunk(x, 2, dim=1) 

        return mean, log_var
    

class VAE_Decoder(nn.Sequential):
    """
    Decoder to reconstruct images from latent space

    Args:
        latent_dim (int): dimension of the latent space
        out_channels (int): number of channels in the output image

    Model Call:
        Input: z: [B, latent_dim, 4, 4] tensor of latent space representation
        Output: [B, out_channels, H, W] tensor of reconstructed images
    """
    def __init__(self, latent_dim=8, out_channels=1):
        super().__init__(
            # Upsample from latent space to (H, W)
            nn.Conv2d(latent_dim, latent_dim, 1, stride=1, padding=0), # (B, latent_dim, H/8, W/8) -> (B, latent_dim, H/8, W/8)
            nn.Conv2d(latent_dim, 512, 3, stride=1, padding=1), # -> (B, 512, H/8, W/8)
            VAEResidualBlock(512, 512),
            VAEAttentionBlock(512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            nn.Upsample(scale_factor=2), # (B, 512, H/8, W/8) -> (B, 512, H/4, W/4)
            nn.Conv2d(512, 512, 3, stride=1, padding=1), # -> (B, 512, H/4, W/4)
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),
            VAEResidualBlock(512, 512),

            nn.Upsample(scale_factor=2), # (B, 512, H/4, W/4) -> (B, 512, H/2, W/2)
            nn.Conv2d(512, 512, 3, stride=1, padding=1), # -> (B, 512, H/2, W/2)
            VAEResidualBlock(512, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),
            VAEResidualBlock(256, 256),

            nn.Upsample(scale_factor=2), # (B, 256, H/2, W/2) -> (B, 256, H, W)
            nn.Conv2d(256, 256, 3, stride=1, padding=1), # -> (B, 256, H, W)
            VAEResidualBlock(256, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),
            VAEResidualBlock(128, 128),

            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, out_channels, 3, stride=1, padding=1), # -> (B, 1, H, W)
        )

    def forward(self, x):
        x /= 0.18215 # remove constant form encoder
        for layer in self:
            x = layer(x)        

        return x


    

    
