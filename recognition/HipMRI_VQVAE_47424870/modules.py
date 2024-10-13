import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Encoder component of VQVAE
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_res_layers, res_hidden_dim):
        super(Encoder, self).__init__()
        self.order = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            ResidualStack(hidden_dim, hidden_dim, res_hidden_dim, n_res_layers)
        )

    def forward(self, x):
        return self.order(x)

class VectorQuantiser(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantiser, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # Embeddings initialized as learnable parameters
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        # Flatten the input spatial dimensions and permute the channels to the last dimension
        b, c, h, w = x.size()
        x_flat = x.view(b, c, -1).transpose(1, 2)  # Shape: [B, H*W, C]

        # Calculate distances between each pixel and the embedding vectors
        # distances: [B, H*W, num_embeddings]
        distances = torch.sum((x_flat.unsqueeze(2) - self.embeddings.unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)

        # Get the indices of the nearest embeddings for each pixel in the flattened spatial dimension
        indices = torch.argmin(distances, dim=2)  # Shape: [B, H*W]

        # Get the quantised vectors using the indices
        z_q = self.embeddings[indices]  # Shape: [B, H*W, embedding_dim]
        
        # Reshape to the original spatial dimensions after transposing
        z_q = z_q.permute(0, 2, 1).view(b, self.embedding_dim, h, w)  # Shape: [B, embedding_dim, H, W]

        # Calculate the quantisation loss
        # Note: x_flat is currently in the shape [B, H*W, C] so needs to be reshaped before loss calculation
        x_reconstructed = z_q.view(b, self.embedding_dim, h * w).transpose(1, 2)  # Shape: [B, H*W, embedding_dim]
        quantisation_loss = F.mse_loss(x_reconstructed.detach(), x_flat) + F.mse_loss(x_reconstructed, x_flat.detach())

        return z_q, quantisation_loss
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_res_layers, res_hidden_dim):
        super(Decoder, self).__init__()
        self.reverse_order = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            ResidualStack(hidden_dim, hidden_dim, res_hidden_dim, n_res_layers),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.reverse_order(x)
    
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim, device):
        super(VQVAE, self).__init__()
        
        # Update the input channels for pre_vq_conv to match the encoder's output
        self.encoder = Encoder(input_dim, hidden_dim)
        self.pre_vq_conv = nn.Conv2d(hidden_dim * 4, embedding_dim, kernel_size=1)  
        self.vq = VectorQuantiser(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim)
        self.to(device)

    def forward(self, x):
        z_e = self.encoder(x)  # Encoding
        z_e = self.pre_vq_conv(z_e)  # Pre-quantisation convolution
        z_q, quantisation_loss = self.vq(z_e)  # Quantisation
        x_reconstructed = self.decoder(z_q)  # Decoding
        return x_reconstructed, quantisation_loss

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