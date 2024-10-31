"""
Components of the VQ-VAE model for generative image reconstruction.

Author: Bairu An, s4702833.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ResidualLayer(nn.Module):
    """A basic residual layer that applies convolutional operations with skip connections."""
    
    def __init__(self, in_channels, out_channels):
        """
        Initialize the residual layer with convolutional operations.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ResidualLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the residual layer."""
        identity = x  # Save the input for the skip connection
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity  # Add the input to the output (skip connection)
        out = self.relu(out)
        return out


class ResidualStack(nn.Module):
    """Stack of residual layers to be used in the encoder and decoder."""
    
    def __init__(self, in_channels, out_channels, num_layers):
        """
        Initialize the stack of residual layers.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_layers (int): Number of residual layers in the stack.
        """
        super(ResidualStack, self).__init__()
        layers = [ResidualLayer(in_channels, out_channels) for _ in range(num_layers)]
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the stack of residual layers."""
        return self.stack(x)


class Encoder(nn.Module):
    """Encoder component of the VQ-VAE model."""
    
    def __init__(self, input_channels, hidden_channels, num_res_layers):
        """
        Initialize the encoder with convolutional layers and residual stacks.
        
        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            num_res_layers (int): Number of residual layers.
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_res_layers)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        """Forward pass through the encoder."""
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """Decoder component of the VQ-VAE model."""
    
    def __init__(self, hidden_channels, output_channels, num_res_layers):
        """
        Initialize the decoder with transposed convolutional layers and residual stacks.
        
        Args:
            hidden_channels (int): Number of hidden channels.
            output_channels (int): Number of output channels.
            num_res_layers (int): Number of residual layers.
        """
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.residual_stack = ResidualStack(hidden_channels, hidden_channels, num_res_layers)
        self.conv2 = nn.ConvTranspose2d(hidden_channels, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        """Forward pass through the decoder."""
        x = self.conv1(x)
        x = self.residual_stack(x)
        x = self.conv2(x)
        return x


class VectorQuantizer(nn.Module):
    """Vector quantization layer for the VQ-VAE model."""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Initialize the vector quantization layer.
        
        Args:
            num_embeddings (int): Number of embeddings in the codebook.
            embedding_dim (int): Dimensionality of each embedding vector.
            commitment_cost (float): Weight for the commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, x):
        """Forward pass through the vector quantization layer."""
        x_flat = x.view(-1, self.embedding_dim)
        distances = (torch.sum(x_flat ** 2, dim=1, keepdim=True) + 
                     torch.sum(self.embeddings.weight ** 2, dim=1) - 
                     2 * torch.matmul(x_flat, self.embeddings.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized_x = self.embeddings(encoding_indices).view(x.shape)

        # Calculate the commitment and quantization losses
        commitment_loss = F.mse_loss(quantized_x, x.detach())
        quantization_loss = F.mse_loss(quantized_x.detach(), x)

        total_loss = quantization_loss + self.commitment_cost * commitment_loss
        quantized_x = x + (quantized_x - x).detach()  # Straight-through estimator
        return quantized_x, total_loss


class VQVAE(nn.Module):
    """Main VQ-VAE model combining the encoder, vector quantizer, and decoder."""
    
    def __init__(self, input_channels, hidden_channels, num_embeddings, embedding_dim, num_res_layers):
        """
        Initialize the VQVAE model.
        
        Args:
            input_channels (int): Number of input channels.
            hidden_channels (int): Number of hidden channels.
            num_embeddings (int): Number of embeddings in the quantizer.
            embedding_dim (int): Dimensionality of each embedding vector.
            num_res_layers (int): Number of residual layers.
        """
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_channels, num_res_layers)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(hidden_channels, input_channels, num_res_layers)

    def forward(self, x):
        """Forward pass through the entire VQ-VAE model."""
        z = self.encoder(x)  # Encode input
        z_q, quantization_loss = self.quantizer(z)  # Quantize latent representation
        x_hat = self.decoder(z_q)  # Decode to reconstruct input
        return x_hat, quantization_loss
