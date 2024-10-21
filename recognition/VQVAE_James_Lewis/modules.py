import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encoder module for VQ-VAE

    Reduces the spatial dimensions of the input tensor
    and increases the number of channels

    @param input_dim: int, number of input channels
    @param dim: int, number of output channels
    @param n_res_block: int, number of residual blocks
    @param n_res_channel: int, number of channels in residual blocks
    @param stride: int, stride of the convolutional layers

    """

    def __init__(self, input_dim, output_dim, n_res_block, n_res_channel, stride):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        self.stride = stride

        self.conv_stack = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, 3, stride, 1),  # First layer: reduce channels
            nn.ReLU(),
            nn.Conv2d(input_dim // 2, input_dim // 4, 3, stride, 1),  # Second layer: further reduce channels
            nn.ReLU(),
            nn.Conv2d(input_dim // 4, input_dim // 4, 3, stride, 1),  # Third layer: maintain channels
            nn.ReLU(),
            nn.Conv2d(input_dim // 4, input_dim // 4, 3, stride, 1),  # Fourth layer: maintain channels
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_stack(x)
        return x

class Decoder(nn.Module):
    """
    Decoder module for VQ-VAE

    Increases the spatial dimensions of the input tensor
    and reduces the number of channels

    @param dim: int, number of input channels
    @param output_dim: int, number of output channels
    @param n_res_block: int, number of residual blocks
    @param n_res_channel: int, number of channels in residual blocks
    @param stride: int, stride of the convolutional layers

    """

    def __init__(self, dim, output_dim, n_res_block, n_res_channel, stride):
        super(Decoder, self).__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        self.stride = stride

        self.inv_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, stride, 1),  # First layer: maintain channels
            nn.ReLU(),
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),  # Second layer: reduce channels, increase spatial size
            nn.ReLU(),
            nn.ConvTranspose2d(dim // 2, output_dim, 4, 2, 1)  # Third layer: output layer to match output dimensions
        )

    def forward(self, x):
        x = self.inv_conv_stack(x)
        return x

class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE

    Discretizes the input tensor and computes the commitment loss

    @param dim: int, number of input channels
    @param n_embed: int, number of embeddings
    @param commitment_cost: float, commitment cost for loss calculation

    """
    def __init__(self, dim, n_embed, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.dim = dim
        self.n_embed = n_embed
        self.commitment_cost = commitment_cost

        self.embed = nn.Embedding(n_embed, dim)
        self.embed.weight.data.uniform_(-1/n_embed, 1/n_embed)

    def forward(self, z):
        #flatten the input tensor
        z_flattened = z.view(-1, z.size(-1))

        #calculate the distances
        distances = (z_flattened ** 2).sum(dim=1, keepdim=True) + \
                    (self.embed.weight ** 2).sum(dim=1) - \
                    2 * torch.matmul(z_flattened, self.embed.weight.t())

        #find the nearest embedding
        min_encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # quantize the input
        z_q = self.embed(min_encoding_indices).view(z.shape)

        #compute the commitment loss
        z_e = z_flattened.view(z.shape)  # Flattened input for loss computation
        commitment_loss = self.commitment_cost * F.mse_loss(z_e, z_q.detach())

        return z_q, min_encoding_indices, commitment_loss


