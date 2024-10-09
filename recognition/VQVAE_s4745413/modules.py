import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Reduces dimensionality of input MRI image
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        self.conv_stack = nn.Sequential(
                nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1), # downsampling
                nn.ReLU(),
                nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
                ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
                )

    def forward(self, x):
        return self.conv_stack(x)

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim: input dimension
    - h_dim: hidden layer dimension
    - res_h_dim: hidden dimension of residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
                )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers:
    - in_dim: input dimension - h_dim: hidden layer dimension
    - res_h_dim: hidden dimension of residual block
    - n_res_layers: number of layers to stack
    """
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList([ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
            x = F.relu(x)
        return x

class Decoder(nn.Module):
    """
    p_phi (x|z) network, given a latent sample z p_phi maps back to original space.
    - in_dim: input dimension
    - h_dim: hidden layer dimensino
    - res_h_dim: hidden dimension of residual block
    - n_res_layers: number of layers to stack
    """
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranpose2d(h_dim//2, 1, kernel_size=4, stride=2, padding=1)
            )
    def forward(self, x):
        return self.inverse_conv_stack(x)

class VectorQuantizer(nn.Module):
    """
    bottleneck part of VQ-VAE
    - n_emb: number of embeddings
    - e_dim: dimension of embedding
    - commit_cost: commitment cost
    """

    def __init__(self, n_emb, e_dim, commit_cost):
        super(VectorQuantizer, self).__init__()
        self.n_emb = n_emb
        self.e_dim = e_dim
        self.commit_cost= commit_cost 

        self.embedding = nn.Embedding(self.n_emb, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_emb, 1.0 / self.n_emb)

    def forward(self, z): # inputs the output of the encoder network into one-hot vector
        batch = 0
        height = 2
        width = 3
        channel = 1
        z = z.permute(batch, height, width, channel).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * \
        torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_idx= torch.argmin(d, dim=1).unsqueeze(1) # find closest encoding
        min_encodings = torch.zeros(min_encoding_idx.shape[0], self.n_emb).to(device)
        min_encodings.scatter_(1, min_encoding_idx, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape) # quantized latent vectors
        # calculate loss
        loss = torch.mean((z_q.detach()-z)**2) + self.commit_cost* torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        z_q = z_q.permute(batch, width, channel, height).contiguous()
        return loss, z_q, perplexity, min_encodings,min_encoding_idx 

