import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual Block Definition
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.activation(out)
        return out

# Self-Attention Layer Definition
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width * height)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, embedding_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels // 2)
        self.conv2 = nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )
        self.attention = SelfAttention(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = self.res_blocks(x)
        x = self.attention(x)
        x = self.bn3(self.conv3(x))
        return x

# decoder
class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, hidden_channels=256, out_channels=1):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.attention = SelfAttention(hidden_channels)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels)
        )
        self.conv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels // 2)
        self.conv3 = nn.ConvTranspose2d(hidden_channels // 2, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = self.attention(x)
        x = self.res_blocks(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = torch.sigmoid(self.conv3(x))
        return x

# Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z):
        # Flatten Input
        z_flattened = z.view(-1, self.embedding_dim)
        
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )
        # Get the nearest embedding index
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(encoding_indices).view(z.shape)

        # Calculate loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = z + (quantized - z).detach()
        return quantized, loss

# VQVAE
class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=256, num_embeddings=1024, embedding_dim=128):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.quantizer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss
