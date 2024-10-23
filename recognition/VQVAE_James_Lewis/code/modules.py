import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_res_block, n_res_channel):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stride = 2

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, output_dim // 2, kernel_size=4, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_dim // 2, output_dim, kernel_size=4, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_dim, 32, kernel_size=3, stride=stride - 1, padding=1),
            ResidualStack(32, 32, n_res_block, n_res_channel)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim, output_dim, n_res_block, n_res_channel):
        super(Decoder, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stride = 1

        self.inv_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(32, output_dim, 3, stride, 1),
            ResidualStack(output_dim, output_dim, n_res_block, n_res_channel),
            nn.ConvTranspose2d(output_dim, output_dim // 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim // 2, 1, 4, 2, 1)  # Output layer for grayscale images
        )

    def forward(self, x):
        x = self.inv_conv_stack(x)
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, dim, n_embed, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        # Initialize embeddings with a smaller range for better stability
        self.embed = nn.Embedding(n_embed, dim)
        self.embed.weight.data.uniform_(-1 / (2 * n_embed), 1 / (2 * n_embed))  # Adjust initialization range

    def forward(self, z):
        # Change the shape of z to [batch_size, height, width, channels]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.dim)

        # Calculate distances between input and embedding weights
        d = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
            (self.embed.weight ** 2).sum(dim=1) - 2 * \
            torch.matmul(z_flattened, self.embed.weight.t())

        # Find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        # Ensure the tensor is on the correct device
        device = z.device
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_embed,
                                    device=device)  # Move tensor to correct device
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embed.weight).view(z.shape)

        # Compute loss for embedding
        loss = (z_q.detach() - z).pow(2).mean() + self.commitment_cost * (z_q - z.detach()).pow(2).mean()

        # Preserve gradients
        z_q = z + (z_q - z).detach()

        # Perplexity
        e_mean = min_encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # Reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, min_encoding_indices, perplexity

class VQVAE(nn.Module):
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

        self.encoder = Encoder(input_dim, dim, n_res_block, n_res_channel)
        self.pre_quantization_conv = nn.Conv2d(
            dim, embedding_dims, kernel_size=1, stride=1)
        # Input dim set to 1 for grayscale
        self.vector_quantizer = VectorQuantizer(embedding_dims, n_embed, commitment_cost)
        self.decoder = Decoder(embedding_dims, dim, n_res_block, n_res_channel)

    def forward(self, x):
        z = self.encoder(x)
        commitment_loss, z_q, min_encoding_indices, perplexity = self.vector_quantizer(z)
        x_recon = self.decoder(z_q)

        return x_recon, commitment_loss



class ResidualStack(nn.Module):
    def __init__(self, in_dim, out_dim, n_res_block, n_res_channel):
        super(ResidualStack, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        stack = []
        for i in range(n_res_block):
            stack.append(nn.Sequential(nn.ReLU(),
                                       nn.Conv2d(in_dim, n_res_channel, 3, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(n_res_channel, out_dim, 1, 1, 0)))
        self.stack = nn.ModuleList(stack)

    def forward(self, x):
        for i in range(self.n_res_block):
            x = self.stack[i](x)
        return x
