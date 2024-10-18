from dataset import load_data, show_example_images
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import utils
import os
from itertools import cycle

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x
    
  
class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - dim_e : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, dim_e, beta=0.25):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.dim_e = dim_e
        self.beta = beta # The paper uses 0.25

        self.embedding = nn.Embedding(self.n_e, self.dim_e)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.dim_e)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        kernel = 4
        stride = 2

        # Adjust the input channels in the encoder from 1 to 64
        self.encoder = nn.Sequential(
            nn.Conv2d(1, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)
        )
        
        self.pre_quant_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        self.codebook = VectorQuantizer(n_embeddings, embedding_dim)
        self.post_quant_conv = nn.ConvTranspose2d(embedding_dim, h_dim, kernel_size=1, stride=1)
        
        # Commitment Loss Beta
        self.beta = 0.25 # From paper
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, h_dim, kernel_size=1, stride=1),  # Adjust to match input[32, 128, 64, 32]
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 1, kernel_size=kernel, stride=stride, padding=1),
        )
        
        self.apply(utils.weights_init)

    def forward(self, x):
        # print(x.shape, "og")
        encoded_output = self.encoder(x)
        # print(encoded_output.shape, "encode")
        encoded_output = self.pre_quant_conv(encoded_output)
        # print(encoded_output.shape, "prequant")
        embedding_loss, quantised_output = self.codebook(encoded_output)
        # print(quantised_output.shape, "quant")
        quantised_output = self.post_quant_conv(quantised_output)
        # print(quantised_output.shape, "postquant")
        decoded_output = self.decoder(quantised_output)
        # print(decoded_output.shape, "decoded")
        
        return decoded_output, embedding_loss, encoded_output, quantised_output
