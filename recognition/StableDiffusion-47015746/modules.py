import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class UNet(nn.Module):
    def __init__(self, latent_dim=128):
        super(UNet, self).__init__()

        # Define downsampling layers (encoder path) with batch normalization
        self.down1 = nn.Conv2d(latent_dim, latent_dim * 2, 3, stride=2, padding=1)  # 128 -> 256 channels, h -> h/2
        self.bn1 = nn.BatchNorm2d(latent_dim * 2)

        self.down2 = nn.Conv2d(latent_dim * 2, latent_dim * 4, 3, stride=2, padding=1)  # 256 -> 512 channels, h/2 -> h/4
        self.bn2 = nn.BatchNorm2d(latent_dim * 4)

        self.down3 = nn.Conv2d(latent_dim * 4, latent_dim * 8, 3, stride=2, padding=1)  # 512 -> 1024 channels, same spatial dimensions
        self.bn3 = nn.BatchNorm2d(latent_dim * 8)

        self.down4 = nn.Conv2d(latent_dim * 8, latent_dim * 16, 3, stride=2, padding=1)  # 1024 -> 2048 channels, same spatial dimensions
        self.bn4 = nn.BatchNorm2d(latent_dim * 16)

        # Define upsampling layers (decoder path) with batch normalization
        self.up4 = nn.ConvTranspose2d(latent_dim * 16, latent_dim * 8, 3, stride=2, padding=1, output_padding=1)  # 2048 -> 1024 channels, h/8 -> h/4
        self.bn5 = nn.BatchNorm2d(latent_dim * 8)

        self.up3 = nn.ConvTranspose2d(latent_dim * 8, latent_dim * 4, 3, stride=2, padding=1, output_padding=1)  # 1024 -> 512 channels, h/4 -> h/2
        self.bn6 = nn.BatchNorm2d(latent_dim * 4)

        self.up2 = nn.ConvTranspose2d(latent_dim * 4, latent_dim * 2, 3, stride=2, padding=1, output_padding=1)  # 512 -> 256 channels, same spatial dimensions
        self.bn7 = nn.BatchNorm2d(latent_dim * 2)

        self.up1 = nn.ConvTranspose2d(latent_dim * 2, latent_dim, 3, stride=2, padding=1, output_padding=1)  # 256 -> 128 channels, same spatial dimensions
        self.bn8 = nn.BatchNorm2d(latent_dim)

    def forward(self, x, t):
        # Downsample (encoder path) with batch normalization
        timestep_embedding = torch.sin(t.float() * 1e-4).view(-1, 1, 1, 1).to(x.device)
    

        x1 = F.relu(self.bn1(self.down1(x + timestep_embedding)))  # Output: [batch, 256, h/2, w/2]
        
        x2 = F.relu(self.bn2(self.down2(x1 + timestep_embedding)))  # Output: [batch, 512, h/4, w/4]
       
        x3 = F.relu(self.bn3(self.down3(x2 + timestep_embedding)))  # Output: [batch, 1024, h/4, w/4]
        
        x4 = F.relu(self.bn4(self.down4(x3 + timestep_embedding)))  # Output: [batch, 2048, h/4, w/4]

        
        # Upsample (decoder path) with skip connections and batch normalization
        x = F.relu(self.bn5(self.up4(x4 + timestep_embedding)))
        
        x = x + x3  # First skip connection
    
        x = F.relu(self.bn6(self.up3(x + timestep_embedding)))
        x = x + x2  # Second skip connection

        x = F.relu(self.bn7(self.up2(x + timestep_embedding)))
        x = F.interpolate(x, size=x1.shape[2:], mode='nearest')

        x = x + x1  # Third skip connection

        x = F.relu(self.bn8(self.up1(x + timestep_embedding)))

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



