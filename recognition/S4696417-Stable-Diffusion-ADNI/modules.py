import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StableDiffusion(nn.Module):
    """
    Main model class

     - Unet predicts noise at each timestep
     - VAE encodes images to latent space and decodes latent representations back to images
     - Noise Scheduler manages the noise shcedule for the diffusion process

    Args:
        unet: UNet model
        vae: VAE model
        noise_scheduler: Noise scheduler

    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Input: t: [B] tensor of timesteps
        Output: [B, C, H, W] Predicted noise tensor
    """
    def __init__(self, unet, vae, noise_scheduler):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler

    def encode(self, x):
        return self.vae.encode(x)
    
    def decode(self, z):
        return self.vae.decode(z)

    def predict_noise(self, z, t):
        pred_noise = self.unet(z, t)
        
        return pred_noise

    def sample(self, num_samples, latent_dim, device, guidance_scale=0.0):
        # Start from pure noise
        z = torch.randn(num_samples, latent_dim, device=device)
        
        for t in reversed(range(self.noise_scheduler.num_timesteps)):
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                pred_noise = self.unet(z, t_batch)
            
            # Remove noise
            z = self.noise_scheduler.remove_noise(z, t_batch, pred_noise)
            
            # Add some noise back (except for the last step)
            if t > 0:
                z += torch.randn_like(z) * self.noise_scheduler.betas[t].sqrt()
        
        # Decode the final latent representation to an image
        with torch.no_grad():
            samples = self.vae.decode(z)
        
        return samples
    
class UNet2(nn.Module):
    """
    Unet to predict noise at each timestep

    Args:
        in_channels: Number of channels in the input image
        hidden_dims: Number of channels in the hidden layers
        time_emb_dim: Number of channels in the time embedding

    Model Call:
        Input: x: [B, C, H, W] noisy image or latent tensor
        Input: t: [B] tensor of timesteps
        Output: [B, C, H, W] Predicted noise tensor

    Unet Structure:
        - Time embedding layer will convert timesteps to vector representation
        - Encoder blocks: Downsampling path with residual and attention blocks
        - Bottleneck: Bottleneck with residual and attention blocks
        - Decoder blocks: Upsampling path with residual and attention blocks
    """
    def __init__(self, in_channels=256, hidden_dims=[128, 256, 512], time_emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Initial linear layer
        self.linear_in = nn.Linear(in_channels, hidden_dims[0])
        
        # Encoder (Downsampling)
        self.down_blocks = nn.ModuleList()
        input_channel = hidden_dims[0]
        for hidden_dim in hidden_dims:
            self.down_blocks.append(ResidualBlock(input_channel, hidden_dim, time_emb_dim))
            input_channel = hidden_dim
        
        # Attention blocks
        self.attn1 = AttentionBlock(hidden_dims[-1])
        self.attn2 = AttentionBlock(hidden_dims[-1])
        
        # Bottleneck
        self.bottleneck1 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        self.bottleneck2 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        
        # Decoder (Upsampling)
        self.up_blocks = nn.ModuleList()
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            self.up_blocks.append(
                ResidualBlock(reversed_hidden_dims[i] * 2, reversed_hidden_dims[i+1], time_emb_dim)
            )
        self.up_blocks.append(ResidualBlock(reversed_hidden_dims[-1] * 2, in_channels, time_emb_dim))
        
        # Final linear layer
        self.linear_out = nn.Linear(in_channels, in_channels)

    def forward(self, x, t):
        # Time embedding
        t = self.time_embed(t)

        # Initial linear layer
        x = self.linear_in(x) # [B, latent_dim] -> [B, hidden_dim]
        
        # Downsampling
        residuals = []
        for down_block in self.down_blocks:
            x = down_block(x, t)
            residuals.append(x)
            x = F.avg_pool2d(x, 2)
        
        # Apply attention
        x = self.attn1(x)
        
        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.bottleneck2(x, t)
        
        # Apply attention
        x = self.attn2(x)
        
        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            residual = residuals.pop()
            x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
            x = torch.cat([x, residual], dim=1)
            x = up_block(x, t)
        
        # Final linear layer
        x = x.squeeze(-1)  
        return self.linear_out(x)
    
class UNet(nn.Module):
    """
    Unet to predict noise at each timestep

    Args:
        in_channels: Number of channels in the input image
        hidden_dims: Number of channels in the hidden layers
        time_emb_dim: Number of channels in the time embedding

    Model Call:
        Input: x: [B, latent_dim, 2, 2] noisy image or latent tensor
        Input: t: [B] tensor of timesteps
        Output: [B, C, H, W] Predicted noise tensor
    """
    def __init__(self, in_channels=256, hidden_dims=[128, 256, 512], time_emb_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Initial conv and linear layer
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, stride=1, padding=1)
        
        # Encoder (Downsampling)
        self.down_blocks = nn.ModuleList()
        input_channel = hidden_dims[0]
        for hidden_dim in hidden_dims:
            self.down_blocks.append(ResidualBlock(input_channel, hidden_dim, time_emb_dim))
            input_channel = hidden_dim
        
        # Attention blocks
        self.attn1 = AttentionBlock(hidden_dims[-1])
        self.attn2 = AttentionBlock(hidden_dims[-1])
        
        # Bottleneck
        self.bottleneck1 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        self.bottleneck2 = ResidualBlock(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        
        # Decoder (Upsampling)
        self.up_blocks = nn.ModuleList()
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i in range(len(reversed_hidden_dims) - 1):
            self.up_blocks.append(
                ResidualBlock(reversed_hidden_dims[i] * 2, reversed_hidden_dims[i + 1], time_emb_dim)
            )
        self.up_blocks.append(ResidualBlock(reversed_hidden_dims[-1] * 2, in_channels, time_emb_dim))
        
        # Final conv layer
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        # Time embedding
        t = self.time_embed(t)
        
        # Initial layers 
        x = self.conv_in(x)
        
        # Downsampling
        residuals = []
        for down_block in self.down_blocks:
            x = down_block(x, t)
            residuals.append(x)
            if x.size(-1) > 1:  # Only downsample if spatial dimensions are > 1
                x = F.avg_pool2d(x, 2)
        
        # Apply attention
        x = self.attn1(x)
        
        # Bottleneck
        x = self.bottleneck1(x, t)
        x = self.bottleneck2(x, t)
        
        # Apply attention
        x = self.attn2(x)
        
        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            if len(residuals) > 0:
                residual = residuals.pop()
                if x.size(-1) < residual.size(-1):
                    x = F.interpolate(x, size=residual.shape[-2:], mode='nearest')
                x = torch.cat([x, residual], dim=1)
            x = up_block(x, t)
        
        # Final linear layer
        return self.conv_out(x)
    

class VAE(nn.Module):
    """
    VAE to compress images to latent space and reconstruct from latent representation
     - Encoder takes input [B, C, H, W] and outputs mean and variance of latent space [B, latent_dim, 4, 4]
     - Decoder takes latent representation [B, latent_dim, 4, 4] and outputs [B, C, H, W] image tensor
     - Noise scheduler manages the noise schedule for the diffusion process
   
    Args:
        in_channels (int): number of channels in the input image
        latent_dim (int): dimension of the latent space
    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Output: [B, C, H, W] Reconstructed image tensor
    """
    def __init__(self, in_channels=1, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
       
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            VAEResidualBlock(32, 64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            VAEResidualBlock(64, 128),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            VAEResidualBlock(128, 256),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv2d(256, latent_dim, 3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(256, latent_dim, 3, stride=1, padding=1)

        self.decoder_input = nn.Conv2d(latent_dim, 256, kernel_size=3, stride=1, padding=1)
       
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            VAEResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            VAEResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            VAEResidualBlock(32, 32),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
   
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.encoder = self.encoder.to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return self

    def encode(self, x):
        x = self.encoder(x)
        mu = self.conv_mu(x)        
        logvar = self.conv_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_input(z)
        output = self.decoder(z)
        return output

    def forward(self, x):
        mu, logvar = self.encode(x)
        latents = self.reparameterize(mu, logvar)
        return self.decode(latents), mu, logvar, latents


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Convert timestep tensor to a vector representation

    Args:
        dim (int): number of channels in the input image

    Model Call:
        Input: time: [B] tensor of timesteps
        Output: [B, dim] vector representation
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """
    Residual convolution block with time embedding for diffusion in Unet

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels in the output image
        time_emb_dim (int): number of channels in the time embedding

    Model Call:
        Input: x: [B, C_in, H, W] tensor of images
        Output: [B, C_out, H, W] processed feature tensor
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        num_groups = min(out_channels, 32)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h += self.time_mlp(t)[:, :, None, None]
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.activation(h + self.shortcut(x))
    
class VAEResidualBlock(nn.Module):
    """
    Res block without time embedding for VAE
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += self.shortcut(residual)
        return F.relu(x)

class AttentionBlock(nn.Module):
    """
    Self Attention block for Unet

    Args:
        channels (int): number of channels in the input image

    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Output: [B, C, H, W] processed feature tensor
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # Flatten the spatial dimensions and transpose to (B, H*W, C)
        x_flat = x.view(b, c, -1).transpose(1, 2)
        
        # Perform self-attention
        x_flat = self.mha(x_flat, x_flat, x_flat)[0]
        
        # Apply layer norm and feedforward network
        x_flat = self.ln(x_flat)
        x_flat = self.ff_self(x_flat)
        
        # Reshape back to original spatial dimensions
        x = x_flat.transpose(1, 2).view(b, c, h, w)
        
        return x

class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
       
        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
       
        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, latents, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(latents.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(latents.device)
       
        # Reshape for broadcasting
        sqrt_alpha_prod = sqrt_alpha_prod.view(-1, 1, 1, 1).expand_as(latents)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1).expand_as(latents)
       
        noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
        return noisy_latents

    def remove_noise(self, noisy_latents, predicted_noise, timesteps):
        alpha_prod = self.alphas_cumprod[timesteps].to(noisy_latents.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(noisy_latents.device)
       
        # Reshape for broadcasting
        alpha_prod = alpha_prod.view(-1, 1, 1, 1).expand_as(noisy_latents)
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.view(-1, 1, 1, 1).expand_as(noisy_latents)
       
        original_latents = (noisy_latents - sqrt_one_minus_alpha_prod * predicted_noise) / torch.sqrt(alpha_prod)
        return original_latents

    def step(self, model_output, timestep, sample):
        prev_timestep = timestep - 1
        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0, device=sample.device)
        beta_prod_t = 1 - alpha_prod_t

        # Reshape for broadcasting
        alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1).expand_as(sample)
        alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1, 1).expand_as(sample)
        beta_prod_t = beta_prod_t.view(-1, 1, 1, 1).expand_as(sample)

        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
       
        # Direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
       
        # x_{t-1}
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
       
        return prev_sample

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        return self

class TimeEmbedding(nn.Module):
    """
    Time embeddings for Unet

    Args:
        dim (int): number of channels in the input image

    Model Call:
        Input: t: [B] tensor of timesteps
        Output: [B, dim]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, t):
        return self.layer(t.unsqueeze(-1).float())