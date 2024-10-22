"""
Author: Farhaan Rashid

Student Number: s4803279

This file contains the source code for the VQVAE model.

Each component of the VQVAE is implemented as a class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block used in the encoder and decoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
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
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying residual connection.
        """
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        if self.projection:
            residual = self.projection(x)

        return self.relu(out + residual)


class Encoder(nn.Module):
    """
    Encoder module for VQVAE-2 with two levels (top and bottom).

    Args:
        in_channels (int): Number of input channels.
        hidden_dims (list): List of hidden dimensions for top and bottom encoders.
        embedding_dims (list): List of embedding dimensions for top and bottom encoders.
    """
    def __init__(self, in_channels, hidden_dims, embedding_dims):
        super(Encoder, self).__init__()

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
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Latent representations from top and bottom encoders (z_top, z_bottom).
        """
        z_top = self.encoder_top(x)
        z_bottom = self.encoder_bottom(x)
        return z_top, z_bottom


class VectorQuantiser(nn.Module):
    """
    Vector Quantiser module for VQVAE.

    Args:
        num_embeddings (int): Number of embedding vectors in the codebook.
        embedding_dim (int): Dimensionality of each embedding vector.
        commitment_cost (float): Weight for the commitment loss.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantiser, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Create the codebook
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)


    def forward(self, inputs):
        """
        Forward pass for vector quantisation.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, embedding_dim, height, width).

        Returns:
            tuple: Quantisation loss, quantised output, and encoding indices.
        """
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

        # Quantise and unflatten
        quantised = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        # Loss
        e_latent_loss = F.mse_loss(quantised.detach(), inputs)
        q_latent_loss = F.mse_loss(quantised, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantised = inputs + (quantised - inputs).detach()

        return loss, quantised, encoding_indices


class Decoder(nn.Module):
    """
    Decoder module for VQVAE-2 with two levels (top and bottom).

    Args:
        in_channels (int): Number of output channels.
        hidden_dims (list): List of hidden dimensions for top and bottom decoders.
        embedding_dims (list): List of embedding dimensions for top and bottom decoders.
    """
    def __init__(self, in_channels, hidden_dims, embedding_dims):
        super(Decoder, self).__init__()

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
        """
        Forward pass of the decoder.

        Args:
            quant_top (torch.Tensor): Quantised top latent representation.
            quant_bottom (torch.Tensor): Quantised bottom latent representation.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        dec_top = self.decoder_top(quant_top)
        dec_top_upsampled = F.interpolate(dec_top, size = quant_bottom.shape[2:])
        combined = torch.cat([dec_top_upsampled, quant_bottom], dim = 1)
        x_recon = self.decoder_bottom(combined)
        return x_recon


class VQVAE2(nn.Module):
    """
    VQVAE-2 model with hierarchical latent spaces.

    Args:
        in_channels (int): Number of input channels.
        hidden_dims (list): List of hidden dimensions for encoder and decoder.
        num_embeddings (list): List of codebook sizes for each level.
        embedding_dim (list): List of embedding dimensions for each level.
        commitment_cost (float): Weight for the commitment loss.
    """
    def __init__(self, in_channels, hidden_dims, num_embeddings, embedding_dims, commitment_cost):
        super(VQVAE2, self).__init__()

        assert len(num_embeddings) == len(embedding_dims)
        self.num_levels = len(num_embeddings)

        # Encoders
        self.encoder = Encoder(in_channels, hidden_dims, embedding_dims)

        # Vector Quantisers
        self.vq_top = VectorQuantiser(num_embeddings[0], embedding_dims[0], commitment_cost)
        self.vq_bottom = VectorQuantiser(num_embeddings[1], embedding_dims[1], commitment_cost)

        # Decoders
        self.decoder = Decoder(in_channels, hidden_dims, embedding_dims)


    def encode(self, x):
        """
        Encodes the input into two latent spaces (top and bottom).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Latent losses and quantised representations for top and bottom levels.
        """
        z_top, z_bottom = self.encoder(x)
        loss_top, quant_top, indices_top = self.vq_top(z_top)
        loss_bottom, quant_bottom, indices_bottom = self.vq_bottom(z_bottom)
        return (loss_top, quant_top, indices_top), (loss_bottom, quant_bottom, indices_bottom)


    def decode(self, quant_top, quant_bottom):
        """
        Decodes the quantised latent representations back to the input space.

        Args:
            quant_top (torch.Tensor): Quantised top latent representation.
            quant_bottom (torch.Tensor): Quantised bottom latent representation.

        Returns:
            torch.Tensor: Reconstructed image.
        """
        return self.decoder(quant_top, quant_bottom)


    def forward(self, x):
        """
        Full forward pass through the VQVAE-2 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Total loss and the reconstructed image.
        """
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
