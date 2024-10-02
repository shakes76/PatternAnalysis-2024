import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, num_residual_hiddens, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        z_e_flattened = z_e.view(-1, self.embedding_dim)
        distances = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_e_flattened, self.embeddings.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view_as(z_e)
        quantized = z_e + (quantized - z_e).detach()
        
        return quantized, encoding_indices

def vq_loss(quantized, z_e, beta=0.25):
    # Reconstruction loss (scaled by beta)
    recon_loss = beta * torch.mean((quantized.detach() - z_e) ** 2)
    
    # Commitment loss (not scaled)
    commitment_loss = torch.mean((quantized - z_e.detach()) ** 2)
    
    # Combine both losses
    return recon_loss + commitment_loss
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, 
                               kernel_size=4, stride=2, padding=1)  # Strided convolution
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 
                               kernel_size=4, stride=2, padding=1)  # Strided convolution
        self.res_block1 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.res_block2 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, 
                               kernel_size=3, stride=1, padding=1)
        self.res_block1 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.res_block2 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.deconv1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 
                                          kernel_size=4, stride=2, padding=1)  # Transposed conv
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(num_hiddens // 2, out_channels=1, 
                                          kernel_size=4, stride=2, padding=1)  # Transposed conv
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x

