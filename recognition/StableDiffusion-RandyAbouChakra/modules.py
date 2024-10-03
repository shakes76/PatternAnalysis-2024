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
