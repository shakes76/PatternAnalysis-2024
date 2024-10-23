"""
Modules for a Vector Quantized Variational Autoencoder (VQVAE) model.

Author: George Reid-Smith

Model Reference:
This model architecture is inspired by the paper:

@article{
    author = {Ali Razavi, Aäron van den Oord, Oriol Vinyals}
    title = {Generating Diverse High-Fidelity Images with VQ-VAE-2}
    year = {2019}
    url = {https://doi.org/10.48550/arXiv.1906.00446}
}

Specifically: https://github.com/google-deepmind/sonnet/tree/v1/sonnet/python/modules/nets
"""

import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    """
    A module that implements the quantization step of the VQVAE.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_embed (int): Number of embedding vectors.
        decay (float): Decay rate for the moving average of embeddings.
        eps (float): Small constant to prevent division by zero.
    """
    
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # Initialize embeddings randomly
        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)  # Fixed embedding matrix
        self.register_buffer("cluster_size", torch.zeros(n_embed))  # Track sizes of clusters
        self.register_buffer("embed_avg", embed.clone())  # Average embedding updates

    def forward(self, input):
        # Flatten the input tensor to compute distances
        flatten = input.reshape(-1, self.dim)

        # Compute distance from each input to each embedding
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        
        # Get indices of the closest embedding
        _, embed_ind = (-dist).max(1)
        
        # One-hot encode the indices
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])  # Reshape back to original dimensions
        quantize = self.embed_code(embed_ind)  # Retrieve corresponding embeddings

        if self.training:
            # Update cluster sizes and embedding averages
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Normalize the embeddings based on cluster sizes
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Compute the difference and detach from the gradient for stable updates
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        """Retrieve the embedding vectors corresponding to the given indices."""
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    """
    A residual block used in the encoder and decoder.

    Attributes:
        in_channel (int): Number of input channels.
        channel (int): Number of output channels.
    """
    
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        # Pass input through the convolutional layers and add the residual
        out = self.conv(input)
        out += input  # Add skip connection
        return out


class Encoder(nn.Module):
    """
    Encoder module of the VQVAE.

    Attributes:
        in_channel (int): Number of input channels.
        channel (int): Number of output channels.
        n_res_block (int): Number of residual blocks to include.
        n_res_channel (int): Number of channels in each residual block.
        stride (int): Stride used in the convolutions.
    """
    
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        # Define convolutional blocks based on stride
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        # Append residual blocks
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # Forward pass through the encoder blocks
        return self.blocks(input)


class Decoder(nn.Module):
    """
    Decoder module of the VQVAE.

    Attributes:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        channel (int): Number of intermediate channels.
        n_res_block (int): Number of residual blocks to include.
        n_res_channel (int): Number of channels in each residual block.
        stride (int): Stride used in the transposed convolutions.
    """
    
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        # Start with a convolution layer
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        # Append residual blocks
        for _ in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # Define transposed convolutions based on stride
        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
                ]
            )
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # Forward pass through the decoder blocks
        return self.blocks(input)


class VQVAE(nn.Module):
    """
    Vector Quantized Variational Autoencoder (VQVAE) model.

    Attributes:
        in_channel (int): Number of input channels (e.g., 1 for grayscale images).
        channel (int): Number of intermediate channels.
        n_res_block (int): Number of residual blocks in the encoder and decoder.
        n_res_channel (int): Number of channels in each residual block.
        embed_dim (int): Dimensionality of the embedding space.
        n_embed (int): Number of embedding vectors in the quantization module.
        decay (float): Decay rate for the embedding update.
    """
    
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        # Define the encoder blocks
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        
        # Quantization for the temporal part of the input
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        
        # Decoder for the temporal part
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        
        # Quantization for the bottleneck
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        
        # Upsampling layer
        self.upsample = nn.ConvTranspose2d(embed_dim, in_channel, 4, stride=2, padding=1)

    def forward(self, input):
        # Encoder forward pass
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        # Quantization step for the temporal part
        quantized_t, diff_t, embed_ind_t = self.quantize_t(self.quantize_conv_t(enc_t))

        # Decoder forward pass for the temporal part
        dec_t = self.dec_t(quantized_t)

        # Prepare input for bottleneck quantization
        dec_b = torch.cat([dec_t, enc_b], dim=1)
        quantized_b, diff_b, embed_ind_b = self.quantize_b(self.quantize_conv_b(dec_b))

        # Final upsampling layer to output
        output = self.upsample(quantized_b)

        # Return the output and the loss components
        loss = diff_t + diff_b
        return output, loss, embed_ind_t, embed_ind_b
