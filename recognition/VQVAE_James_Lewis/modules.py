import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):
    """
    Encoder module for VQ-VAE

    Reduces the spatial dimensions of the input tensor
    and increases the number of channels

    @param input_dim: int, number of input channels
    @param dim: int, number of output channels
    @param n_res_block: int, number of residual blocks
    @param n_res_channel: int, number of channels in residual blocks
    @param stride: int, stride of the convolutional layers

    """

    def __init__(self, input_dim, output_dim, n_res_block, n_res_channel):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, output_dim//2, kernel_size=4, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_dim//2, output_dim, kernel_size=4, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_dim, 32, kernel_size=3, stride=stride - 1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):

        x = self.conv_stack(x)
        return x

class Decoder(nn.Module):
    """
    Decoder module for VQ-VAE

    Increases the spatial dimensions of the input tensor
    and reduces the number of channels

    @param dim: int, number of input channels
    @param output_dim: int, number of output channels
    @param n_res_block: int, number of residual blocks
    @param n_res_channel: int, number of channels in residual blocks
    @param stride: int, stride of the convolutional layers

    """

    def __init__(self, dim, output_dim, n_res_block, n_res_channel):
        super(Decoder, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stride = 1

        self.inv_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(32, output_dim, 3, stride, 1),  # First layer: maintain channels
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim, output_dim // 2, 4, 2, 1),  # Second layer: reduce channels, increase spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim // 2, 3, 4, 2, 1)  # Third layer: output layer to match output dimensions
        )

    def forward(self, x):
        x = self.inv_conv_stack(x)
        return x

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE

    Discretizes the input tensor and computes the commitment loss

    @param dim: int, number of input channels
    @param n_embed: int, number of embeddings
    @param commitment_cost: float, commitment cost for loss calculation

    """
    def __init__(self, dim, n_embed, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.embed = nn.Embedding(n_embed, dim)
        self.embed.weight.data.uniform_(-1/n_embed, 1/n_embed)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.dim)


        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embed.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embed.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_embed).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embed.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

class VQVAE(nn.Module):
    """
    VQ-VAE module

    Combines the encoder, decoder and vector quantizer modules

    @param input_dim: int, number of input channels
    @param dim: int, number of output channels
    @param n_res_block: int, number of residual blocks
    @param n_res_channel: int, number of channels in residual blocks
    @param stride: int, stride of the convolutional layers
    @param n_embed: int, number of embeddings
    @param commitment_cost: float, commitment cost for loss calculation

    """

    def __init__(self, input_dim, dim, n_res_block, n_res_channel, stride, n_embed, commitment_cost, embedding_dims):
        super(VQVAE, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        self.stride = stride
        self.embedding_dims = embedding_dims
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.encoder = Encoder(3, dim, n_res_block, n_res_channel)
        self.vector_quantizer = VectorQuantizer(embedding_dims, n_embed, commitment_cost)
        self.decoder = Decoder(embedding_dims, dim, n_res_block, n_res_channel)


    def forward(self, x):
        z = self.encoder(x)
        commitment_loss, z_q, perplexity, min_encodings, min_encoding_indices,  = self.vector_quantizer(z)
        x_recon = self.decoder(z_q)

        return x_recon, commitment_loss

