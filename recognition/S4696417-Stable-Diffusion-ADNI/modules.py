import torch, wandb, math
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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

    @torch.no_grad()
    def sample(self, epoch, shape, device='cuda'):
        x = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(self.noise_scheduler.num_timesteps)), desc="Sampling"):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t)
            x = self.noise_scheduler.step(pred_noise, t, x)

        final_image = self.vae.decode(x)
        wandb.log({f"sample at epoch {epoch}": wandb.Image(final_image)})
        return x
    

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
            #self.down_blocks.append(ResidualBlock(input_channel, hidden_dim, time_emb_dim))
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


class NoiseScheduler_Depricated:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule_type='linear'):
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        self.betas = self._cosine_beta_schedule(num_timesteps)

        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, x, t, predicted_noise):
        alpha_prod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        return (beta_prod_t.sqrt() * predicted_noise - x * beta_prod_t) / alpha_prod_t.sqrt()

    def step(self, model_output, timesteps, sample):
        # Handle batched timesteps
        t = timesteps
        prev_t = (t - 1).clamp(min=0)
        
        pred_original_sample = self._predict_x0_from_eps(sample, t, model_output)
        
        alpha_prod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t].view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_sample_direction = (1 - alpha_prod_t_prev) * model_output
        
        # Compute x_{t-1} mean
        pred_prev_sample = (
            alpha_prod_t_prev.sqrt() * pred_original_sample +
            pred_sample_direction
        )
        
        # Add noise
        variance = ((beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)).sqrt()
        noise = torch.randn_like(model_output)
        variance = variance.view(-1, 1, 1, 1) * noise
        
        # Only add variance where t > 0
        mask = (t > 0).float().view(-1, 1, 1, 1)
        pred_prev_sample = pred_prev_sample + variance * mask
        
        return pred_prev_sample

    def _predict_x0_from_eps(self, x_t, t, eps):
        alpha_prod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (x_t - (1 - alpha_prod_t).sqrt() * eps) / alpha_prod_t.sqrt()

    def to(self, device):
        # Move all tensors in the class to the specified device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self


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


class NoiseScheduler_Depricated:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        self.betas = self._cosine_beta_schedule(num_timesteps)
        
        # Define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def _cosine_beta_schedule(self, timesteps):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def remove_noise(self, noisy_latent, timesteps, noise_pred):
        alpha_prod = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        pred_original_sample = (noisy_latent - sqrt_one_minus_alpha_prod * noise_pred) / torch.sqrt(alpha_prod)
        return pred_original_sample

    def step(self, model_output, timesteps, sample):
        t = timesteps
        prev_t = (t - 1).clamp(min=0)
        
        alpha_prod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prod_t_prev = self.alphas_cumprod[prev_t].view(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        pred_original_sample = self.remove_noise(sample, t, model_output)
        
        pred_sample_direction = (1 - alpha_prod_t_prev) * model_output
        
        # Compute x_{t-1} mean
        pred_prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample +
            pred_sample_direction
        )
        
        # Add noise
        variance = ((beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)).sqrt()
        noise = torch.randn_like(model_output)
        pred_prev_sample += variance.view(-1, 1, 1, 1) * noise * (t > 0).float().view(-1, 1, 1, 1)
        
        return pred_prev_sample
    
    def to(self, device):
        # Move all tensors in the class to the specified device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self
    



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
        B, seq_len, d_embed = in_shape
        interim_shape = (B, seq_len, self.num_heads, self.d_head)
        
        q, k, v = self.in_proj(x).chunk(3, dim=-1) # (B, seq_len, d_embed) -> 3 x (B, seq_len, dim)
        q = q.view(interim_shape).transpose(1, 2) # (B, seq_len, dim) -> (B, num_heads, seq_len, dim / num_heads)
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


class NoiseScheduler:
    """
    Noise scheduler based on Fast-DDPM implementation in stable diffusion
    """
    def __init__(self, num_timesteps=10, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    def add_noise(self, x, noise, t):
        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x +
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
        )

    def step(self, model_output, timestep, sample):
        """
        Step the noise schduler back one step

        Args:
            model_output (torch.Tensor): model output
            timestep (torch.Tensor): timestep
            sample (torch.Tensor): sample
        """
        t = timestep
        prev_t = (t - 1).clamp(min=0)

        alpha = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[prev_t]

        beta = 1- alpha / alpha_prev

        # Ensure proper broadcasting
        beta = beta.view(-1, 1, 1, 1)
        alpha = alpha.view(-1, 1, 1, 1)
        alpha_prev = alpha_prev.view(-1, 1, 1, 1)

        pred_original_sample = (sample - beta.sqrt() * model_output) / alpha.sqrt()
        pred_sample_direction = (1 - alpha_prev).sqrt() * model_output
        prev_sample = alpha_prev.sqrt() * pred_original_sample + pred_sample_direction

        noise = torch.randn_like(sample)
        variance = ((1 - alpha_prev) / (1 - alpha) * beta).sqrt()
        prev_sample = prev_sample + variance * noise * (t > 0).float().view(-1, 1, 1, 1)

        return prev_sample

    def to(self, device):
        # Move all tensors in the class to the specified device
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
        return self
