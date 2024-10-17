import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # B, 64, 128, 128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # B, 128, 64, 64
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # B, 256, 32, 32
            nn.ReLU(),
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        return latent

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # B, 128, 64, 64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # B, 64, 128, 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), # B, 3, 256, 256
            nn.Tanh(),
        )
        
    def forward(self, x):
        reconstruction = self.decoder(x)
        return reconstruction

def forward_diffusion(x_0, timesteps, beta):
    noise = torch.randn_like(x_0)
    x_t = x_0
    for t in range(timesteps):
        x_t = (1 - beta[t]) * x_t + (beta[t] ** 0.5) * noise
    return x_t, noise
    
def reverse_diffusion(x_t, timesteps, beta):
    for t in reversed(range(timesteps)):
        x_t = x_t / (1 - beta[t])  # Denoising step
    return x_t

