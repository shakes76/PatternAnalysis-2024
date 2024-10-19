# Contains the source code of the GFNet Vision Transformer

import torch
import torch.nn as nn
import torch.fft as fft


class GFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 num_classes=2, num_layers=6, dropout=0.1):
        """
        GFNet-inspired Vision Transformer for image classification using Fourier transforms.
        Args:
            img_size (int): Input image size (assumes square images).
            patch_size (int): Size of each image patch.
            in_channels (int): Number of input channels (3 for RGB).
            embed_dim (int): Dimension of the patch embeddings.
            num_classes (int): Number of output classes (2 for AD vs. NC).
            num_layers (int): Number of Fourier-based transformer layers.
            dropout (float): Dropout rate.
        """
        super(GFNet, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )  # Converts image to patch embeddings

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Feedforward layers applied after Fourier transform
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Stacking multiple Fourier-based layers
        self.layers = nn.ModuleList([self._build_layer(embed_dim) for _ in range(num_layers)])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def _build_layer(self, embed_dim):
        """
        Helper function to create a single Fourier-based transformer block.
        """
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            FourierTransformLayer(embed_dim),  # Custom Fourier transform layer
            self.ffn
        )

    def forward(self, x):
        # Create patch embeddings
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through Fourier-based transformer layers
        for layer in self.layers:
            x = layer(x)

        # Take the mean of all patch embeddings (global average pooling)
        x = self.norm(x.mean(dim=1))

        # Classification
        logits = self.fc(x)
        return logits


class FourierTransformLayer(nn.Module):
    def __init__(self, embed_dim):
        """
        Layer that applies a Fourier Transform to each input patch embedding.
        Args:
            embed_dim (int): Dimension of the input embeddings.
        """
        super(FourierTransformLayer, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, x):
        # Apply 1D Fourier transform along the embedding dimension
        x_fft = fft.fft(x, dim=-1).real
        return x_fft



