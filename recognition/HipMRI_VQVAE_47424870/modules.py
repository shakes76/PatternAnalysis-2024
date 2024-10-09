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

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        # Flatten the input
        b, c, h, w = x.size()
        x = x.view(b, c, -1).transpose(1, 2)  # [B, H*W, C]

        # Calculate distances
        distances = (x.unsqueeze(2) - self.embeddings.unsqueeze(0).unsqueeze(0)) ** 2
        distances = distances.sum(dim=1)  # [B, H*W, num_embeddings]
        
        # Get the nearest embeddings
        indices = distances.argmin(dim=2)  # [B, H*W]
        z_q = self.embeddings[indices]  # [B, H*W, embedding_dim]
        
        # Reshape to original dimensions
        z_q = z_q.transpose(1, 2).view(b, self.embedding_dim, h, w)

        # Calculate the quantization loss
        quantization_loss = F.mse_loss(z_q.detach(), x) + F.mse_loss(z_q, x.detach())

        return z_q, quantization_loss
    

# Define other components: Decoder, VQVAE.
