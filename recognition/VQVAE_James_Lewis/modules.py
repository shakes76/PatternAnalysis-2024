import torch
import torch.nn as nn


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

    def __init__(self, input_dim, dim, n_res_block, n_res_channel, stride):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.n_res_block = n_res_block
        self.n_res_channel = n_res_channel
        self.stride = stride

        self.conv_stack = nn.Sequential(
            nn.Conv2d(input_dim, dim, 3, stride, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, stride, 1),
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
            nn.ConvTranspose2d(dim, dim, 3, stride, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(dim, output_dim, 3, stride, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim, output_dim, 3, stride, 1)
        )

    def forward(self, x):
        x = self.inv_conv_stack(x)
        return x