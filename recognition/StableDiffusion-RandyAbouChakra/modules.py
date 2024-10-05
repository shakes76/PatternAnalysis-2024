import torch
import torch.nn as nn

class SinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_positions=10000):
        super(SinusoidalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        position = torch.arange(0, max_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_positions, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, timestep):
        return self.pe[timestep]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Sinusoidal embedding for timesteps
        self.time_embedding = SinusoidalEmbedding(embedding_dim=512)

        # Encoder (downsampling) path
        self.down1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)

        # Decoder (upsampling) path
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

        # Activation
        self.relu = nn.ReLU()

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x, timestep):
        # Get timestep embedding
        time_emb = self.time_embedding(timestep).view(-1, 512, 1, 1)  # Reshape to match input shape
        
        # Encoder
        x1 = self.relu(self.bn1(self.down1(x)))  # 64 x 32 x 32
        x2 = self.relu(self.bn2(self.down2(x1))) # 128 x 16 x 16
        x3 = self.relu(self.bn3(self.down3(x2))) # 256 x 8 x 8
        x4 = self.relu(self.bn4(self.down4(x3))) # 512 x 4 x 4

        # Add timestep embedding to the bottleneck
        x4 = x4 + time_emb

        # Decoder
        x = self.relu(self.up1(x4))              # 256 x 8 x 8
        x = self.relu(self.up2(x + x3))          # Skip connection from x3
        x = self.relu(self.up3(x + x2))          # Skip connection from x2
        x = self.up4(x)                          # 3 x 64 x 64

        return x

class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))
        return x
    

class LatentDiffusionModel(nn.Module):
    def __init__(self, encoder, decoder, unet, timesteps=1000):
        super(LatentDiffusionModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.unet = unet
        self.timesteps = timesteps
        self.betas = self._linear_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, 0)

    def _linear_beta_schedule(self, timesteps):
        return torch.linspace(0.0001, 0.02, timesteps)

    def forward_diffusion(self, z, t):
        """
        Add noise to the latent z space according to a variance schedule.
        """
        noise = torch.randn_like(z)
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        return torch.sqrt(alpha_hat_t) * z + torch.sqrt(1 - alpha_hat_t) * noise, noise

    def reverse_denoise(self, z_t, t):
        """
        Apply the reverse process to denoise the latent code.
        """
        pred_noise = self.unet(z_t, t)
        return pred_noise

    def forward(self, x, t):
        """
        Full forward pass from image to noise prediction.
        """
        # Encode image to latent space
        z = self.encoder(x)
        
        # Apply forward diffusion in latent space
        z_noisy, noise = self.forward_diffusion(z, t)
        
        # Predict noise added in latent space
        pred_noise = self.reverse_denoise(z_noisy, t)
        
        return pred_noise, noise

    def decode_latent(self, z):
        """
        Decode latent space back to image.
        """
        return self.decoder(z)