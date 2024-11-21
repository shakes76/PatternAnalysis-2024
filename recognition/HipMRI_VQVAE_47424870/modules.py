import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2

# Define the Encoder component of VQVAE
class Encoder(nn.Module):
    """
    Encoder component of the VQVAE, used to progressively downsample
    the input images through convolutional layers and apply residual stacks.

    Args:
        in_channels (int): Number of input channels in the input data.
        hidden_channels (int): Number of channels for hidden layers.
        res_channels (int): Number of channels in residual layers.
        nb_res_layers (int): Number of residual layers in the residual stack.
        downscale_factor (int): Factor by which the spatial dimensions are downscaled.

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

    Returns:
        torch.Tensor: Output tensor after encoding.
    """
    def __init__(self, in_channels, hidden_channels, res_channels, nb_res_layers, downscale_factor):
        super(Encoder, self).__init__()
        assert log2(downscale_factor) % 1 == 0, "Downscale must be a power of 2"
        downscale_steps = int(log2(downscale_factor))

        # Encoder layers for downsampling
        layers = []
        c_channel, n_channel = in_channels, hidden_channels // 2
        for _ in range(downscale_steps):
            # Each layer halves the spatial dimensions with stride 2
            layers.append(nn.Sequential(
                nn.Conv2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, hidden_channels

        # Final convolutional layer and residual stack
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))
        layers.append(ResidualStack(n_channel, res_channels, nb_res_layers))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Define Vector Quantiser (VQ) component
class VectorQuantiser(nn.Module):
    """
    Vector Quantiser component of VQVAE, mapping input embeddings to nearest
    prototype embeddings using vector quantisation.

    Args:
        in_channels (int): Number of input channels in data.
        embed_dim (int): Dimension of the embedding space.
        nb_entries (int): Number of embedding entries in the codebook.

    Forward Args:
        x (torch.Tensor): Input tensor after encoding, of shape [batch, channels, height, width].

    Returns:
        tuple:
            - torch.Tensor: Quantised tensor of shape [batch, channels, height, width].
            - torch.Tensor: Vector quantisation loss.
            - torch.Tensor: Embedding indices.
    """
    def __init__(self, in_channels, embed_dim, nb_entries):
        super(VectorQuantiser, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, embed_dim, 1)  # Project input channels to embedding dimension

        # Initialize embedding parameters
        self.dim = embed_dim
        self.n_embed = nb_entries
        self.decay = 0.99  # Decay rate for moving average
        self.eps = 1e-5  # Small constant to avoid division by zero

        # Create embeddings, cluster size, and moving average of embeddings
        embed = torch.randn(embed_dim, nb_entries, dtype=torch.float32)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(nb_entries, dtype=torch.float32))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):
        # Flatten and compute distances between inputs and embeddings
        x = self.conv_in(x.float()).permute(0, 2, 3, 1)
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        
        # One-hot encoding of embedding indices
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        
        # Quantize input by embedding index
        quantize = self.embed_code(embed_ind)

        # EMA update for embeddings
        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # Compute VQ loss (difference between input and quantized representation)
        diff = (quantize.detach() - x).pow(2).mean()
        quantize = x + (quantize - x).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    # Convert embedding indices to their corresponding vectors
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))
    
# Define the Decoder component of VQVAE
class Decoder(nn.Module):
    """
    Decoder component of the VQVAE, used to progressively upsample the quantised
    embeddings through convolutional layers and apply residual stacks.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels in the decoding layers.
        out_channels (int): Number of output channels.
        res_channels (int): Number of channels in residual layers.
        nb_res_layers (int): Number of residual layers in residual stack.
        upscale_factor (int): Factor by which spatial dimensions are upscaled.

    Forward Args:
        x (torch.Tensor): Quantised tensor from VectorQuantiser.

    Returns:
        torch.Tensor: Decoded tensor of shape [batch, channels, height, width].
    """
    def __init__(self, in_channels, hidden_channels, out_channels, res_channels, nb_res_layers, upscale_factor):
        super(Decoder, self).__init__()
        assert log2(upscale_factor) % 1 == 0, "Upscale must be a power of 2"
        upscale_steps = int(log2(upscale_factor))

        # Decoder layers for upsampling
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, stride=1, padding=1)]
        layers.append(ResidualStack(hidden_channels, res_channels, nb_res_layers))
        c_channel, n_channel = hidden_channels, hidden_channels // 2
        for _ in range(upscale_steps):
            # Each layer doubles the spatial dimensions with stride 2
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(c_channel, n_channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(n_channel),
                nn.ReLU(inplace=True),
            ))
            c_channel, n_channel = n_channel, out_channels

        # Final convolutional layer
        layers.append(nn.Conv2d(c_channel, n_channel, 3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(n_channel))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
# Define the VQVAE model
class VQVAE(nn.Module):
    """
    Vector Quantised Variational Autoencoder (VQVAE) model combining encoder, vector quantiser,
    and decoder components for generative modeling.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Hidden layer channels in encoder/decoder.
        res_channels (int): Residual layer channels.
        nb_res_layers (int): Number of residual layers in encoder/decoder.
        nb_levels (int): Number of quantisation levels.
        embed_dim (int): Dimension of embeddings.
        nb_entries (int): Number of embedding vectors.
        scaling_rates (list): Down/upscaling rates for each VQVAE level.

    Forward Args:
        x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

    Returns:
        tuple:
            - torch.Tensor: Output decoded tensor.
            - list: Vector quantisation losses from each quantisation level.
    """
    def __init__(self, in_channels=3, hidden_channels=128, res_channels=32, nb_res_layers=2, nb_levels=3, embed_dim=64, nb_entries=512, scaling_rates=[8,4,2]):
        super(VQVAE, self).__init__()
        self.nb_levels = nb_levels

        # Create encoders for each level
        self.encoders = nn.ModuleList([Encoder(in_channels, hidden_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for sr in scaling_rates[1:]:
            self.encoders.append(Encoder(hidden_channels, hidden_channels, res_channels, nb_res_layers, sr))

        # Create codebooks for each level
        self.codebooks = nn.ModuleList()
        for i in range(nb_levels - 1):
            self.codebooks.append(VectorQuantiser(hidden_channels + embed_dim, embed_dim, nb_entries))
        self.codebooks.append(VectorQuantiser(hidden_channels, embed_dim, nb_entries))

        # Create decoders for each level
        self.decoders = nn.ModuleList([Decoder(embed_dim * nb_levels, hidden_channels, in_channels, res_channels, nb_res_layers, scaling_rates[0])])
        for sr in scaling_rates[1:]:
            self.decoders.append(Decoder(embed_dim * (nb_levels - 1), hidden_channels, embed_dim, res_channels, nb_res_layers, sr))

    def forward(self, x):
        encoder_outputs = []
        code_outputs = []
        diffs = []

        # Pass input through encoders
        for enc in self.encoders:
            if len(encoder_outputs):
                encoder_outputs.append(enc(encoder_outputs[-1]))
            else:
                encoder_outputs.append(enc(x))

        # Quantise encoder outputs through codebooks
        for l in range(self.nb_levels - 1, -1, -1):
            codebook = self.codebooks[l]
            if len(code_outputs):
                code_resized = F.interpolate(code_outputs[-1], size=encoder_outputs[l].shape[2:])
                code_q, diff, _ = codebook(torch.cat([encoder_outputs[l], code_resized], dim=1))
            else:
                code_q, diff, _ = codebook(encoder_outputs[l])

            diffs.append(diff)
            code_outputs.append(code_q)

        # Concatenate codebook outputs and decode
        target_size = code_outputs[-1].shape[2:]  
        code_outputs_resized = [F.interpolate(c, size=target_size) for c in code_outputs]
        decoder_output = self.decoders[0](torch.cat(code_outputs_resized, dim=1))
        return decoder_output, diffs

# Residual layer used in encoder/decoder
class ResidualLayer(nn.Module):
    """
    Residual layer with adjustable residual weight for encoding/decoding stability.

    Args:
        in_channels (int): Input channel size.
        res_channels (int): Channel size for intermediate layers in residual connections.

    Forward Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after residual connection.
    """
    def __init__(self, in_channels, res_channels):
        super(ResidualLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_channels, in_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.layers(x) * self.alpha + x
    
# Stack of residual layers
class ResidualStack(nn.Module):
    """
    Stack of residual layers used in encoder/decoder for improved feature extraction.

    Args:
        in_channels (int): Input channel size.
        res_channels (int): Channel size for intermediate layers in residual connections.
        nb_layers (int): Number of residual layers in the stack.

    Forward Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor after residual layers are applied.
    """
    def __init__(self, in_channels, res_channels, nb_layers):
        super(ResidualStack, self).__init__()
        self.stack = nn.Sequential(*[ResidualLayer(in_channels, res_channels) for _ in range(nb_layers)])

    def forward(self, x):
        return self.stack(x)
