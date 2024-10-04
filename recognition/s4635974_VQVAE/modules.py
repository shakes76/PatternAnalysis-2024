import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_residual_hiddens, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Skip connection

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, dim_embedding, beta):
        super(VectorQuantizer, self).__init__()
        self.dim_embedding = dim_embedding
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, dim_embedding)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.beta = beta
    
    def forward(self, z_e):
        print("Quantized z_e type:", z_e.dtype)
        z_e_flattened = z_e.view(-1, self.dim_embedding)
        distances = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_e_flattened, self.embeddings.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view_as(z_e)
        print("Quantized shape:", quantized.shape)
        quantized = z_e + (quantized - z_e).detach()
        print("Quantized tensor type:", quantized.dtype)

        # Calculate loss
        recon_loss = F.mse_loss(quantized.detach(), z_e)
        # Commitment loss (scaled)
        commitment_loss = self.beta * F.mse_loss(quantized, z_e.detach()) ###
        loss = recon_loss + commitment_loss

        return loss, quantized, encoding_indices
    
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
        print("Decoder input type before processing:", x.dtype)
        print("Decoder input shape before processing:", x.shape)
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x
    
class VQVAE(nn.Module):
    def __init__(self, num_channels, num_hiddens, num_residual_hiddens,
                 num_embeddings, dim_embedding, beta) -> None:
        super(VQVAE, self).__init__()
        self.encoder = Encoder(num_channels,num_hiddens, num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=dim_embedding, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(num_embeddings, dim_embedding, beta)
        self.decoder = Decoder(dim_embedding, num_hiddens, num_residual_hiddens)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss, quantized, _ = self.vq(x)
        x = self.decoder(quantized)

        return loss, x

