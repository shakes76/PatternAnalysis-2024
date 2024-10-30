# Containing the source code of the components of your model. 
# Each component must be implementated as a class or a function

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLayer(nn.Module):
    """A basic residual layer."""
    def __init__(self, in_channels, out_channels):
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Skip connection
        out = self.relu(out)
        return out


class ResidualStack(nn.Module):
    """Stack of residual layers."""
    def __init__(self, in_channels, out_channels, num_layers):
        super(ResidualStack, self).__init__()
        layers = [ResidualLayer(in_channels, out_channels) for _ in range(num_layers)]
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)


class Encoder(nn.Module):
    """Encoder for the VQVAE model."""
    def __init__(self, input_channels, hidden_channels, num_res_layers):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_res_layers)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder for the VQVAE model."""
    def __init__(self, hidden_channels, output_channels, num_res_layers):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_res_layers)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv2(x)
        return x


class VectorQuantizer(nn.Module):
    """Vector quantization layer for the VQVAE."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        x_flat = x.view(-1, self.embedding_dim)
        distances = (torch.sum(x_flat ** 2, dim=1, keepdim=True) + 
                     torch.sum(self.embeddings.weight ** 2, dim=1) - 
                     2 * torch.matmul(x_flat, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized_x = self.embeddings(encoding_indices).view(x.shape)

        # Calculate the loss
        commitment_loss = F.mse_loss(quantized_x, x.detach())
        quantization_loss = F.mse_loss(quantized_x.detach(), x)

        total_loss = quantization_loss + self.commitment_cost * commitment_loss
        quantized_x = x + (quantized_x - x).detach()  # Straight-through estimator
        return quantized_x, total_loss


class VQVAE(nn.Module):
    """The main VQVAE model combining the encoder, quantizer, and decoder."""
    def __init__(self, input_channels, hidden_channels, num_embeddings, embedding_dim, num_res_layers):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, num_res_layers)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(hidden_channels, input_channels, num_res_layers)

    def forward(self, x):
        z = self.encoder(x)
        z_q, quantization_loss = self.quantizer(z)
        x_hat = self.decoder(z_q)
        return x_hat, quantization_loss
