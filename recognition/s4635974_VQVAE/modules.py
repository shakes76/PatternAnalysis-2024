"""
COMP3710 VQ-VAE report
Author: David Collie
Date: October, 2024

VQ-VAE Model Implementation

This file contains the implementation of the VQ-VAE (Vector Quantized Variational Autoencoder) model.
It includes all key components: the encoder, vector quantizer, and decoder.

The VQ-VAE model is composed of:
- `ResidualBlock`: Implements a residual block with skip connections to enhance feature learning.
- `Encoder`: A convolutional encoder that down-samples the input using convolutions and residual blocks.
- `VectorQuantizer`: Responsible for quantizing the encoded features to discrete latent embeddings.
- `Decoder`: A transposed convolutional decoder that reconstructs the input from quantized embeddings.
- `VQVAE`: Integrates the encoder, vector quantizer, and decoder into a full VQ-VAE model, handling both 
the forward pass and loss computation.

This modular design allows the VQ-VAE to efficiently encode images, quantize the latent representations, 
and decode them, while minimizing reconstruction loss and commitment loss.
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels for the second convolution.
        num_residual_hiddens (int): Number of output channels for the first convolution.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True), 
            nn.Conv2d(in_channels, num_residual_hiddens, 
                      kernel_size=3, stride=1, padding=1, bias=False), # 3x3 conv
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, 
                      kernel_size=1, stride=1, bias=False) # 1x1 conv
        )
    
    def forward(self, x):
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Output tensor with skip connection applied.
        """
        return x + self.block(x)  # Skip connection

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer layer for VQ-VAE, responsible for mapping inputs to discrete latent embeddings.

    Args:
        num_embeddings (int): Number of embedding vectors.
        dim_embedding (int): Dimension of each embedding vector.
        beta (float): Weighting for the commitment loss.
    """
    def __init__(self, num_embeddings, dim_embedding, beta):
        super(VectorQuantizer, self).__init__()
        self.dim_embedding = dim_embedding
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Embedding(num_embeddings, dim_embedding) # Embedding layer
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings) # Initialize embeddings
        self.beta = beta # Commitment loss weighting
    
    def forward(self, z_e):
        """
        Forward pass of the vector quantizer.
        
        Args:
            z_e (torch.Tensor): Input tensor from the encoder of shape (N, C, H, W).
        
        Returns:
            loss (torch.Tensor): Total loss combining reconstruction and commitment loss.
            quantized (torch.Tensor): Quantized tensor with same shape as z_e.
            encoding_indices (torch.Tensor): Indices of selected embeddings.
        """
        z_e_flattened = z_e.view(-1, self.dim_embedding) # Flatten input for distance calculation
        # Compute distances between embeddings and input vectors
        distances = torch.sum(z_e_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embeddings.weight ** 2, dim=1) - \
                    2 * torch.matmul(z_e_flattened, self.embeddings.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1) # Get nearest embedding index
        quantized = self.embeddings(encoding_indices).view_as(z_e) # Get quantized output
        quantized = z_e + (quantized - z_e).detach() # Pass-through mechanism for gradients

        # Calculate loss
        recon_loss = F.mse_loss(quantized.detach(), z_e) # Reconstruction loss
        # Commitment loss (scaled)
        commitment_loss = self.beta * F.mse_loss(quantized, z_e.detach()) # Scaled commitment loss
        loss = recon_loss + commitment_loss

        return loss, quantized, encoding_indices
    
class Encoder(nn.Module):
    """
    Encoder for VQ-VAE, consisting of two strided convolutions followed by residual blocks.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden units (output channels) in convolutions.
        num_residual_hiddens (int): Number of hidden units in residual blocks.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens // 2, 
                               kernel_size=4, stride=2, padding=1)  # First downsampling conv
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_hiddens // 2, num_hiddens, 
                               kernel_size=4, stride=2, padding=1)  # Second downsampling conv
        self.res_block1 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.res_block2 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block

    def forward(self, x):
        """
        Forward pass of the encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Encoded feature map.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x

class Decoder(nn.Module):
    """
    Decoder for VQ-VAE, consisting of two residual blocks followed by two transposed convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden units (output channels) in convolutions.
        num_residual_hiddens (int): Number of hidden units in residual blocks.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_hiddens, 
                               kernel_size=3, stride=1, padding=1) # First conv
        self.res_block1 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.res_block2 = ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)  # Residual block
        self.deconv1 = nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, 
                                          kernel_size=4, stride=2, padding=1)  # First transposed conv
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(num_hiddens // 2, out_channels=1, 
                                          kernel_size=4, stride=2, padding=1)  # Second transposed conv
    
    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            torch.Tensor: Reconstructed image.
        """
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x
    
class VQVAE(nn.Module):
    """
    VQ-VAE model, which includes encoder, vector quantizer, and decoder.
    
    Args:
        num_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden units in encoder and decoder.
        num_residual_hiddens (int): Number of hidden units in residual blocks.
        num_embeddings (int): Number of embedding vectors in vector quantizer.
        dim_embedding (int): Dimensionality of each embedding vector.
        beta (float): Weighting for the commitment loss.
    """
    def __init__(self, num_channels, num_hiddens, num_residual_hiddens,
                 num_embeddings, dim_embedding, beta) -> None:
        super(VQVAE, self).__init__()
        self.encoder = Encoder(num_channels,num_hiddens, num_residual_hiddens)
        self.pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, out_channels=dim_embedding, kernel_size=1, stride=1)
        self.vq = VectorQuantizer(num_embeddings, dim_embedding, beta)
        self.decoder = Decoder(dim_embedding, num_hiddens, num_residual_hiddens)

    def forward(self, x):
        """
        Forward pass of the VQ-VAE model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W).
        
        Returns:
            loss (torch.Tensor): VQ-VAE loss (reconstruction + commitment).
            torch.Tensor: Reconstructed output of shape (N, C, H, W).
        """
        x = self.encoder(x)
        x = self.pre_vq_conv(x)
        loss, quantized, _ = self.vq(x)
        x = self.decoder(quantized)

        return loss, x

