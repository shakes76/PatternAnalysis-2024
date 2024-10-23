"""
Modules for a Vector Quantized Variational Autoencoder (VQVAE) model.

Author: George Reid-Smith

Model Reference:
This model architecture is borrowed from the paper:

@article{
    citation = {1}
    author = {Ali Razavi, Aäron van den Oord, Oriol Vinyals}
    title = {Generating Diverse High-Fidelity Images with VQ-VAE-2}
    year = {2019}
    url = {https://doi.org/10.48550/arXiv.1906.00446}
}

Specifically: https://github.com/google-deepmind/sonnet/tree/v1/sonnet/python/modules/nets
"""

import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    """Vector quantization module for discretizing latent vectors.

    Performs vector quantization by finding closest embedding in a codebook 
    to the input latent vector. Maintains exponential moving average of the codebook 
    adaptation during training.
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # Codebook (embedding vectors)
        embed = torch.randn(dim, n_embed)

        # Track code book, exponential moving average of codebook, and cluster sizes
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        """Performs vector quantization on input latent vector.

        Args:
            input (torch.Tensor): input latent vector

        Returns:
            (tuple): tuple containing quantized latent vector, quantization loss, and indices of 
            nearest codebook entries
        """

        # Flatten input
        flatten = input.reshape(-1, self.dim)
        
        # Compute distances between input and codebook entries
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        # Find closest codebook entries for each latent vector
        _, embed_ind = (-dist).max(1)

        # Convert indices to one-hot encodings for embedding lookup
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        
        # Quantize latent vectors by looking up nearest codebook
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            # Update cluster sizes and codebook using exponential moving average
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Calculate quantization loss
        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind
    
    def embed_code(self, embed_id):
        """Looks up codebook entries for given indicies.

        Args:
            embed_id (torch.Tensor): tensor containing indices of codebook

        Returns:
            torch.Tensor: tensor containing corresponding entries
        """
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    

class ResBlock(nn.Module):
    """Residual block for the encoder and decoder.
    """
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        """Forward pass of the residual block.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        out = self.conv(input)
        out += input
        return out
    

class Encoder(nn.Module):
    """Encoder for VQVAE.

    Encodes the input image tensor into quanitzed latent representations.
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()
        
        # Define convolutions for given strides
        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        # Append residual blocks
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)


    def forward(self, input):
        """Forward pass of the encoder.

        Args:
            input (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor (latent representation)
        """
        return self.blocks(input)
    

class Decoder(nn.Module):
    """Decoder for VQVAE.

    Constructs a reconstructed image tensor using the quantized latent representations 
    generated by the encoder.
    """
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))
        blocks.append(nn.ReLU(inplace=True))

        # Define deconvolutions for given strides
        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        """Forward pass of the decoder.

        Args:
            input (torch.Tensor): input tensor (quantized latent representation)

        Returns:
            torch.Tensor: reconstruction of original input tensor
        """
        return self.blocks(input)


class VQVAE(nn.Module):
    """VQVAE model.

    Takes an input tensor, encodes into a latent representation and quantizes 
    the representation. Decodes to reconstruct original input tensor.
    
    References: [1]
    """
    def __init__(
        self,
        in_channel=1,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()
        # Encoder modules for processing input into latent representation
        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        
        # Quantization model for discretizng latent representations
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)

        # Decoder modules for reconstruction input from quantized representation
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        """Forward pass of the VQVAE model.

        Args:
            input (torch.Tensor): input image tensor

        Returns:
            torch.Tensor: reconstruction of original image tensor
        """
        # Encode into quantized latent representaton
        quant_t, quant_b, diff, _, _ = self.encode(input)

        # Decode to reconstruct original input tensor
        dec = self.decode(quant_t, quant_b)

        return dec, diff
    
    def encode(self, input):
        """Encode input tensor into a quantized latent representation.

        Args:
            input (torch.Tensor): input image tensor

        Returns:
            tuple: quantized top and bottom latent representation, 
            quantization loss, and indices of nearest codebook embeddings 
            for both top and bottom.
        """

        # Pass through bottom encoder
        enc_b = self.enc_b(input)

        # Pass encoded output through the top encoder
        enc_t = self.enc_t(enc_b)

        # Quantize the top latent representation
        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        # Decode the quantized representation to feed into bottom encoder
        dec_t = self.dec_t(quant_t)
        
        # Combine decoded top representation with bottom encoder output
        enc_b = torch.cat([dec_t, enc_b], 1)

        # Quantize bottom latent representation
        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b
    
    def decode(self, quant_t, quant_b):
        """Decode quantized latent representation

        Args:
            quant_t (torch.Tensor): top quantized latent representation
            quant_b (torch.Tensor): bottom quantized latent representation

        Returns:
            torch.Tensor: reconstruction of original image tensor
        """
        # Upsample quantized top representation
        upsample_t = self.upsample_t(quant_t)

        # Combine upsampled top representation with quantized bottom representation
        quant = torch.cat([upsample_t, quant_b], 1)

        # Pass combined representation through decoder to reconstruct original image tensor
        dec = self.dec(quant)

        return dec
    
    def decode_code(self, code_t, code_b):
        """Decode given code indicies into an input tensor.

        Args:
            code_t (torch.Tensor): codebook indicies for top latent representation
            code_b (torch.Tensor): codebook indicies for bottom latent representation

        Returns:
            torch.Tensor: Reconstructed input tensor
        """
        # Codebook lookups
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        # Decode representations into a reconstructed input image tensor
        dec = self.decode(quant_t, quant_b)

        return dec