import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Sinusoidal Embedding for timestep
class SinusoidalPositionEmbeddings(nn.Module):
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
# Noise scheduler for adding noise to latent representations at various timesteps
class NoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
        self.alphas = torch.cumprod(1 - self.betas, dim=0).to(device)

    def add_noise(self, x, t):
        noise = torch.randn_like(x).to(device)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1).to(x.device)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise, noise

# Define a class for a residual network block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb=32):
        super(ResNetBlock, self).__init__()
        self.time_mlp =  nn.Linear(time_emb, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential() # Shortcut for identity mapping
        if in_channels != out_channels:
            # Adjust dimensions if necessary
            self.shortcut.add_module('conv_shortcut', nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
            self.shortcut.add_module('bn_shortcut', nn.BatchNorm2d(out_channels))

    def forward(self, x, t):
        residual = self.shortcut(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        time_emb = self.time_mlp(t)
        time_emb = self.relu(time_emb)
        time_emb = time_emb[(..., ) + (None,) * 2]
        x = x + time_emb
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        x += residual
        x = self.relu(x)
        
        return x

# Define a class for an encoder block
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, num_blocks=2, time_emb=32):
        super().__init__()
        self.blocks = nn.ModuleList([ResNetBlock(in_c if i == 0 else out_c, out_c) for i in range(num_blocks)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        for block in self.blocks:
            x = block(x, t)
        skip = x.clone() # Keep a copy for skip connection
        x = self.pool(x)
        return x, skip

# Define a class for a decoder block
class DecoderBlock(nn.Module):
    def __init__(self, num_in, num_out, num_blocks=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(num_in, num_out, kernel_size=2, stride=2, padding=0)
        self.blocks = nn.ModuleList([ResNetBlock(num_out * 2 if i == 0 else num_out, num_out) for i in range(num_blocks)])

    def forward(self, x, t, skip):
        # Upsample `x` to match the spatial dimensions of `skip`
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='nearest')  # Resize `x` to match `skip`

        # Concatenate with the skip connection
        x = torch.cat([x, skip], axis=1)

        # Pass through the blocks
        for block in self.blocks:
            x = block(x, t)
        return x

# Define the main class for the diffusion network
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder blocks
        self.down1 = EncoderBlock(128, 256)
        self.down2 = EncoderBlock(256, 512)
        self.down3 = EncoderBlock(512, 1024)
        
        # Bottleneck block
        self.bottle_neck = ResNetBlock(1024, 2048)
        
        # Decoder blocks
        self.up1 = DecoderBlock(2048, 1024)
        self.up2 = DecoderBlock(1024, 512)
        self.up3 = DecoderBlock(512, 256)

        # Batch normalization and output layer
        self.norm_out = nn.BatchNorm2d(256) 
        self.out = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        
        # Time embedding layers
        time_dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

    def forward(self, x, t):
        t = self.time_mlp(t)
        
        residuals = []  # Keep the skip connections
        
        # Downsample
       
        x, skip1 = self.down1(x, t)
        residuals.append(skip1)
        
        x, skip2 = self.down2(x, t)
        residuals.append(skip2)
        
        x, skip3 = self.down3(x, t)
        residuals.append(skip3)
        
        
        # Bottleneck
        x = self.bottle_neck(x, t)
        
        # Upsample
        x = self.up1(x, t, residuals.pop())
       
        x = self.up2(x, t, residuals.pop())
        
        x = self.up3(x, t, residuals.pop())
        
        # Normalisation
        x = self.norm_out(x)
        
        return self.out(x)
        
# Encoder: reduces the image to a latent representation
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.layers(x)

# Decoder: reconstructs the image from the latent representation
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1),
            nn.Tanh()  # To keep the output within [-1, 1] range
        )

    def forward(self, x):
        return self.layers(x)
    
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, out_channels=3):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim).to(device)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=out_channels).to(device)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
def check_grad(tensor, name):
    if tensor.requires_grad:
        print(f"{name}: requires_grad=True, grad_fn={tensor.grad_fn}")
    else:
        print(f"{name}: requires_grad=False")


class DiffusionModel(nn.Module):
    def __init__(self, encoder, unet, decoder, noise_scheduler):
        super(DiffusionModel, self).__init__()
        self.encoder = encoder
        self.unet = unet
        self.decoder = decoder
        self.noise_scheduler = noise_scheduler

        # Set requires_grad=False for all decoder parameters
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, x, t):
        # Encode to latent space without gradients
        with torch.no_grad():
            latent = self.encoder(x)

        # Add noise
        noisy_latent, noise = self.noise_scheduler.add_noise(latent, t)

        # Denoise with UNet (this requires gradients)
        predicted_noise = self.unet(noisy_latent, t)

        return latent, noisy_latent, predicted_noise, noise



