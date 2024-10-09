# modules.py

import torch
from torch import nn
from torch.nn import functional as F

class ResidualStack(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        # Residual blocks that add identity mapping for deeper networks. This helps
        # the model better capture the important features of the MRI data
        layers = []
        for i in range(num_residual_layers):
            layers.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_hiddens,
                        out_channels=num_residual_hiddens,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=num_residual_hiddens,
                        out_channels=num_hiddens,
                        stride=1,
                        kernel_size=1,
                    ),
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        h = x
        for layer in self.layers:
            # Adds identity mapping (skip connection)
            h = h + layer(h)
        return torch.relu(h)
    
class Encoder(nn.Module):
    def __init__(
            self,
            in_channels,
            num_hiddens, 
            num_downsampling_layers, 
            num_residual_layers, 
            num_residual_hiddens
    ):
        super().__init__()
        conv = nn.Sequential()
        for downsampling_layer in range(num_downsampling_layers):
            if downsampling_layer == 0:
                out_channels = num_hiddens // 2
            elif downsampling_layer == 1:
                (in_channels, out_channels) = (num_hiddens // 2, num_hiddens)
            else:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)

            conv.add_module(
                f"down{downsampling_layer}",
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4, # Try switch to 3 to see the affects 
                    stride=2, # Stride 2 downsamples by half 
                    padding=1
                ),
            )
            conv.add_module(f"relu{downsampling_layer}", nn.ReLU())
            
        conv.add_module(
            "final_conv",
            nn.Conv2d(
                in_channels=num_hiddens,
                out_channels=num_hiddens,
                kernel_size=3,
                padding=1
            ),
        )
        self.conv = conv
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        h = self.conv(x)
        return self.residual_stack(h)
    
class Decoder(nn.Module):
    def __init__(
            self,
            embedding_dim, # Will be set to 64, but try 128 if images are not clear enough
            num_hiddens,
            num_upsampling_layers,
            num_residual_layers,
            num_residual_hiddens
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_hiddens,
            kernel_size=3,
            padding=1,
        )
        self.residual_stack = ResidualStack(
            num_hiddens, num_residual_layers, num_residual_hiddens
        )
        upconv = nn.Sequential()
        for upsampling_layer in range(num_upsampling_layers):
            if upsampling_layer < num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens)
            elif upsampling_layer == num_upsampling_layers - 2:
                (in_channels, out_channels) = (num_hiddens, num_hiddens // 2)
            else:
                (in_channels, out_channels) = (num_hiddens // 2, 1) # 1 for grayscale images 

            upconv.add_module(
                f"up{upsampling_layer}",
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2, # Stride 2 upsamples by double
                    padding=1
                ),
            )
            if upsampling_layer < num_upsampling_layers - 1:
                upconv.add_module(f"relu{upsampling_layer}", nn.ReLU())

        self.upconv = upconv

    def forward(self, x):
        h = self.conv(x)
        h = self.residual_stack(h)
        x_recon = self.upconv(h)
        return x_recon
    
class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.epsilon = epsilon

        # Initialises embeddings randomly
        limit = 3 ** 0.5
        self.embeddings = nn.Parameter(torch.FloatTensor(embedding_dim, num_embeddings).uniform_(-limit, limit))

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

        # Computes distances to embedding vectors to find the closest match to the input
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.embeddings
            + (self.embeddings ** 2).sum(0, keepdim=True)
        )

        # Gets the closest embedding index
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.embeddings.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        # Calculates the commitment loss
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()

        # Straight-through estimator
        quantized_x = x + (quantized_x - x).detach()

        return quantized_x, commitment_loss, encoding_indices.view(x.shape[0], -1)

class VQVAE(nn.Module):
    def __init__(
            self, 
            in_channels=1, # Grayscale MRI slices
            num_hiddens=128,
            num_downsampling_layers=3, # Adjust this to see if it improves generated images (256x256 -> 32x32)
            num_residual_layers=2,
            num_residual_hiddens=32,
            embedding_dim=64,
            num_embeddings=256,
            decay=0.99,
            epsilon=1e-5
    ):
        super().__init__()
        self.encoder = Encoder(
            in_channels,
            num_hiddens,
            num_downsampling_layers,
            num_residual_layers,
            num_residual_hiddens
        )
        self.pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1
        )
        self.vq = VectorQuantizer(
            embedding_dim, num_embeddings, decay, epsilon
        )
        self.decoder = Decoder(
            embedding_dim,
            num_hiddens,
            num_upsampling_layers=num_downsampling_layers, # Should match downsampling layers
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens
        )

    def forward(self, x):
        z = self.pre_vq_conv(self.encoder(x))
        z_quantized, commitment_loss, _ = self.vq(z)
        x_recon = self.decoder(z_quantized)
        return {"commitment_loss": commitment_loss, "x_recon": x_recon}

