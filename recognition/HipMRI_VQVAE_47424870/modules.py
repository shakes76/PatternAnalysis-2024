import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Encoder component of VQVAE
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

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
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(embedding_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x
    
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_embeddings, embedding_dim):
        super(VQVAE, self).__init__()
        
        # Update the input channels for pre_vq_conv to match the encoder's output
        self.encoder = Encoder(input_dim, hidden_dim)
        self.pre_vq_conv = nn.Conv2d(hidden_dim * 4, embedding_dim, kernel_size=1)  
        self.vq = VectorQuantiser(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim)

    def forward(self, x):
        z_e = self.encoder(x)  # Encoding
        z_e = self.pre_vq_conv(z_e)  # Pre-quantisation convolution
        z_q, quantisation_loss = self.vq(z_e)  # Quantisation
        x_reconstructed = self.decoder(z_q)  # Decoding
        return x_reconstructed, quantisation_loss

