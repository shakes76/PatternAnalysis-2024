import torch, wandb, math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler

class StableDiffusion(nn.Module):
    """
    Stable diffusion model. 
    Used to train noise prediction and generate new samples from diffusion process

     - Unet predicts noise at each timestep
     - VAE encodes images to latent space and decodes latent representations back to images
     - Noise Scheduler manages the noise shcedule for the diffusion process

    Args:
        unet: UNet model
        vae: VAE model
        noise_scheduler: Noise scheduler
        image_size: Size of the image

    Methods:
        encode: Encode images to latent space
        decode: Decode latent representations back to images
        sample_latent: Sample latent representations
        predict_noise: Predict noise at each timestep
        sample: Generate samples from the model
    """
    def __init__(self, unet, vae, noise_scheduler, image_size):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.image_size = image_size

    def encode(self, x):
        return self.vae.encode(x)
    
    def decode(self, z):
        return self.vae.decode(z)
    
    def sample_latent(self, mu, logvar):
        return self.vae.sample(mu, logvar)

    def predict_noise(self, z, t):
        pred_noise = self.unet(z, t)
        return pred_noise

    @torch.no_grad()
    def sample(self, num_images, device='cuda'):
        """
        Generate new images from the diffusion model and logs to wandb

        Args:
            num_images: Number of images to generate
            device: Device to run the model on

        Returns:
            final_image: Generated images
        """
        shape = (num_images, self.vae.latent_dim, int(self.image_size/8), int(self.image_size/8))
        x = torch.randn(shape, device=device)
        steps = reversed(range(self.noise_scheduler.num_timesteps))
        for i in tqdm(steps, desc="Sampling"):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.noise_scheduler.step(self.unet, x, t)

        final_image = self.vae.decode(x)
        wandb.log({f"sample": wandb.Image(final_image)})
        return final_image
    

class UNet(nn.Module):
    """
    Unet to predict noise at each timestep

    Args:
        in_channels: Number of channels in the input image
        hidden_dims: Number of channels in the hidden layers
        time_emb_dim: Number of channels in the time embedding

    Methods:
        forward: Predict noise at timestep t
    """
    def __init__(self, in_channels=8, hidden_dims=[128, 256, 512], time_emb_dim=256):
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
    

class VAEResidualBlock(nn.Module):
    """
    Residual block for VAE encoder/decoder

    Args:
        in_channels (int): number of channels in the input image
        out_channels (int): number of channels in the output image
    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Output: [B, C, H, W] Reconstructed image tensor
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, x):
        residual = x

        x = self.group_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.group_norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        x += self.shortcut(residual)

        return x


class VAEAttentionBlock(nn.Module):
    """
    Attention block for VAE encoder/decoder

    Args:
        channels (int): number of channels in the input image
    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Output: [B, C, H, W] Reconstructed image tensor
    """
    def __init__(self, channels):
        super().__init__()

        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        
        residual = x
        x = self.group_norm(x)

        # reshape to (B, C, H*W)
        B, C, H, W = x.shape
        x = x.view((B, C, H * W)) 
        x = x.transpose(-1, -2) # (B, C, H*W) -> (B, H*W, C)

        x = self.attention(x)
        x = x.transpose(-1, -2) # (B, H*W, C) -> (B, C, H*W)
        x = x.view((B, C, H, W)) # (B, C, H*W) -> (B, C, H, W)

        x += residual

        return x


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
    

class SelfAttention(nn.Module):
    """
    Self Attention block for VAE encoder/decoder

    Args:
        num_heads (int): number of heads
        d_embed (int): number of channels in the input image

    Model Call:
        Input: x: [B, C, H, W] tensor of images
        Output: [B, C, H, W] Reconstructed image tensor
    """
    def __init__(self, num_heads, d_embed, in_bias=True, out_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_bias)
        self.num_heads = num_heads
        self.d_head = d_embed // num_heads


    def forward(self, x):
        in_shape = x.shape
        B, seq_len, _ = in_shape
        interim_shape = (B, seq_len, self.num_heads, self.d_head)
        
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # (B, seq_len) -> 3 x (B, seq_len, dim)
        q = q.view(interim_shape).transpose(1, 2) # -> (B, num_heads, seq_len, dim / num_heads)
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 

        # (B, num_heads, seq_len, dim / num_heads) @ (B, num_heads, dim / num_heads, seq_len) -> (B, num_heads, seq_len, seq_len)
        weight = q @ k.transpose(-2, -1) 
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, num_heads, seq_len, seq_len) @ (B, num_heads, seq_len, dim / num_heads) -> (B, num_heads, seq_len, dim / num_heads)
        output = weight @ v
        output = output.transpose(1, 2) #  -> (B, seq_len, num_heads, dim / num_heads)
        output = output.reshape(in_shape) #  -> (B, seq_len, dim)
        output = self.out_proj(output) 

        return output


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
    

class TimeEmbedding(nn.Module):
    """
    Time embeddings for Unet

    Args:
        n_emb (int): numberr of time embeddings
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


class CosineAnnealingWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(CosineAnnealingWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [self.min_lr + (base_lr - self.min_lr) * 
                    (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs]
        

class NoiseScheduler_Fast_DDPM():
    """
    Noise scheduler based on the Fast-DDPM paper (https://arxiv.org/abs/2405.14802)
    Improved performance with fewer iterations compared to DDPM

    Args:
        num_timesteps (int): number of timesteps
        beta_start (float): initial value of beta
        beta_end (float): final value of beta

    Methods:
        step(model, noisy_latents, timesteps): Perform one step of the diffusion process
        add_noise(x_0, noise, t): Add noise to the image    
        fast_sampling(model, shape, device, num_steps): Generate samples from the model
        to(device): Move all tensors in the class to the specified device
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # Calculate alpha values
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculate diffusion coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        # Calculate posterior variance
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def add_noise(self, x_0, noise, t):
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def step(self, model, noisy_latents, timesteps):
        # Ensure timesteps is a 1D tensor
        timesteps = timesteps.view(-1)
        
        # Get the corresponding beta and alpha values
        betas = self.betas[timesteps]
        alphas = self.alphas[timesteps]
        alphas_cumprod = self.alphas_cumprod[timesteps]
        
        # Expand dimensions for broadcasting
        betas = betas.view(-1, 1, 1, 1)
        alphas = alphas.view(-1, 1, 1, 1)
        alphas_cumprod = alphas_cumprod.view(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = model(noisy_latents, timesteps)
        
        # Calculate mean
        mean = (noisy_latents - betas * predicted_noise / torch.sqrt(1 - alphas_cumprod)) / torch.sqrt(alphas)
        
        # Calculate variance
        variance = betas * (1 - alphas_cumprod) / (1 - alphas_cumprod)
        
        # Add noise only if not at the last step
        noise = torch.randn_like(noisy_latents)
        mask = (timesteps > 0).float().view(-1, 1, 1, 1)
        
        denoised_latents = mean + mask * torch.sqrt(variance) * noise
        
        return denoised_latents

    def fast_sampling(self, model, shape, device, num_steps=50):
        x_t = torch.randn(shape, device=device)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps).long().to(device)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x_t = self.step(model, x_t, t_batch)
        
        return x_t
    
    def to(self, device):
        # Move all tensors in the class to the specified device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self
    
