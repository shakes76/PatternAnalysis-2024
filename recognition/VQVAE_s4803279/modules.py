"""
This file contains the source code for the VQVAE model.

Each component of the VQVAE is implemented as a class.
"""
"""
regarding the model, can i just use the structure that i find in papers and online resources?
    can use a premade model just understand how it works and augment the data in my own way.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block used in the encoder and decoder
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding = 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding = 1)
        self.relu = nn.ReLU(inplace = False)

        if in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.projection = None


    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.projection:
            residual = self.projection(x)

        return self.relu(out + residual)


class EncoderTopBottom(nn.Module):
    def __init__(self, in_channels, hidden_dims, embedding_dims):
        super(EncoderTopBottom, self).__init__()

        # Top encoder
        self.encoder_top = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], kernel_size = 4, stride = 2, padding = 1),
            ResidualBlock(hidden_dims[0], hidden_dims[0]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[0], embedding_dims[0], 1)
        )

        # Bottom encoder
        self.encoder_bottom = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[1], kernel_size = 4, stride = 2, padding = 1),
            ResidualBlock(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[1], embedding_dims[1], 1)
        )

    def forward(self, x):
        z_top = self.encoder_top(x)
        z_bottom = self.encoder_bottom(x)
        return z_top, z_bottom


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQVAE

    Args:
        num_embeddings: size of the codebook
        embedding_dim: dimension of each codebook vector
        commitment_cost: weight for commitment loss
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Create the codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)


    def forward(self, inputs):
        # Ensure the embeddings are on the same device as inputs
        self.embedding = self.embedding.to(inputs.device)

        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim = 1, keepdim = True)
                    + torch.sum(self.embedding.weight ** 2, dim = 1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim = 1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device = inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized, encoding_indices


class DecoderTopBottom(nn.Module):
    def __init__(self, in_channels, hidden_dims, embedding_dims):
        super(DecoderTopBottom, self).__init__()

        # Top decoder
        self.decoder_top = nn.Sequential(
            nn.ConvTranspose2d(embedding_dims[0], hidden_dims[0], kernel_size = 4, stride = 2, padding = 1),
            ResidualBlock(hidden_dims[0], hidden_dims[0]),
            nn.ReLU()
        )

        # Bottom decoder
        self.decoder_bottom = nn.Sequential(
            nn.ConvTranspose2d(embedding_dims[1] + hidden_dims[0], hidden_dims[1], kernel_size = 4, stride = 2, padding = 1),
            ResidualBlock(hidden_dims[1], hidden_dims[1]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims[1], in_channels, 1)
        )

    def forward(self, quant_top, quant_bottom):
        dec_top = self.decoder_top(quant_top)
        dec_top_upsampled = F.interpolate(dec_top, size = quant_bottom.shape[2:])
        combined = torch.cat([dec_top_upsampled, quant_bottom], dim = 1)
        x_recon = self.decoder_bottom(combined)
        return x_recon


class VQVAE2(nn.Module):
    """
    VQVAE-2 with hierarchical latent spaces

    Args:
        in_channels: number of input channels
        hidden_dims: list of hidden dimensions for encoder/decoder
        num_embeddings: list of codebook sizes for each level
        embedding_dim: list of embedding dimensions for each level
        commitment_cost: weight for commitment loss
    """
    def __init__(self, in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost):
        super(VQVAE2, self).__init__()

        assert len(num_embeddings) == len(embedding_dims)
        self.num_levels = len(num_embeddings)

        # Encoders
        self.encoder = EncoderTopBottom(in_channels, hidden_dims, embedding_dims)

        # Vector Quantizers
        self.vq_top = VectorQuantizer(num_embeddings[0], embedding_dims[0], commitment_cost)
        self.vq_bottom = VectorQuantizer(num_embeddings[1], embedding_dims[1], commitment_cost)

        # Decoders
        self.decoder = DecoderTopBottom(in_channels, hidden_dims, embedding_dims)


    def encode(self, x):
        z_top, z_bottom = self.encoder(x)
        loss_top, quant_top, indices_top = self.vq_top(z_top)
        loss_bottom, quant_bottom, indices_bottom = self.vq_bottom(z_bottom)
        return (loss_top, quant_top, indices_top), (loss_bottom, quant_bottom, indices_bottom)


    def decode(self, quant_top, quant_bottom):
        return self.decoder(quant_top, quant_bottom)


    def forward(self, x):
        # Move input to the same device as the model
        device = next(self.parameters()).device
        x = x.to(device)

        # Encode
        (loss_top, quant_top, _), (loss_bottom, quant_bottom, _) = self.encode(x)

        # Decode
        x_recon = self.decode(quant_top, quant_bottom)

        # Calculate reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Total loss
        total_loss = recon_loss + loss_top + loss_bottom

        return total_loss, x_recon
