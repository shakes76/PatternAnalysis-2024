import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Noise scheduler for adding noise to latent representations at various timesteps
class NoiseScheduler:
    def __init__(self, timesteps=1000):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = torch.cumprod(1 - self.betas, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

# UNet: Processes data in latent space with skip connections (simplified for demonstration)
class UNet(nn.Module):
    def __init__(self, latent_dim=128):
        super(UNet, self).__init__()
        # Define downsampling and upsampling layers
        self.down1 = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.down2 = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.up1 = nn.ConvTranspose2d(latent_dim, latent_dim, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(latent_dim, latent_dim, 3, padding=1)

    def forward(self, x, t):
        # Add a simple time embedding to encoded features
        timestep_embedding = torch.sin(t.float() * 1e-4).view(-1, 1, 1, 1)
        x = x + timestep_embedding  # Inject time embedding

        # Downsample (encode path)
        x1 = F.relu(self.down1(x))
        x2 = F.relu(self.down2(x1))

        # Upsample (decode path with skip connection)
        x = F.relu(self.up1(x2)) + x1
        x = F.relu(self.up2(x))
        return x

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
    

# Main StableDiffusionModel class that integrates all components
class StableDiffusionModel(nn.Module):
    def __init__(self, latent_dim=128, timesteps=1000):
        super(StableDiffusionModel, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.unet = UNet(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.noise_scheduler = NoiseScheduler(timesteps=timesteps)
        self.timesteps = timesteps

    def forward(self, x, t):
        # Encode image to latent space
        latent = self.encoder(x)
        # Add noise to the latent representation
        noisy_latent = self.noise_scheduler.add_noise(latent, t)
        # Denoise the noisy latent representation with UNet
        denoised_latent = self.unet(noisy_latent, t)
        # Decode latent space back to image
        return latent, noisy_latent, denoised_latent, self.decoder(denoised_latent)