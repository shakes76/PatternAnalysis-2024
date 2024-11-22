import torch
import torch.nn as nn
import torch.fft
from functools import partial
from torch.nn.init import trunc_normal_
from timm.models.layers import DropPath
import math


class Percep(nn.Module):
    """A perceptron module with two linear layers and dropout for feed-forward operations in Transformer blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super(Percep, self).__init__()

        # Set output features to input features if not specified
        out_features = out_features or in_features

        # Set hidden features to input features if not specified
        hidden_features = hidden_features or in_features

        # First linear layer
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Activation layer
        self.act = act_layer()

        # Second linear layer
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Apply first linear transformation
        x = self.fc1(x)

        # Apply activation
        x = self.act(x)

        # Apply dropout
        x = self.drop(x)

        # Apply second linear transformation
        x = self.fc2(x)

        # Apply dropout again
        x = self.drop(x)

        return x


class GlobalFilter(nn.Module):
    """Applies a global filter in the frequency domain using complex weights and FFT for efficient spatial processing."""

    def __init__(self, dim, h=14, w=8):
        super(GlobalFilter, self).__init__()

        # Initialize complex weights
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

        # Store width and height of the spatial dimensions
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        # Get batch size, number of tokens, and channel dimension
        B, N, C = x.shape

        # Calculate square root of N to find spatial dimensions if spatial_size is not given
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        # Reshape input tensor to match spatial dimensions
        x = x.view(B, a, b, C).to(torch.float32)

        # Apply FFT with conjugate symmetry to reduce computation (as per paper)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Convert complex weight parameter to complex type
        weight = torch.view_as_complex(self.complex_weight)

        # Apply filter in the frequency domain
        x = x * weight

        # Inverse FFT to return to spatial domain
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        # Reshape back to original dimensions
        x = x.reshape(B, N, C)

        return x


class Block(nn.Module):
    """A Transformer block with a GlobalFilter and perceptron layer for processing patches."""

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super(Block, self).__init__()

        # Normalization layer before the filter
        self.norm1 = norm_layer(dim)

        # Global filter layer for spatial processing
        self.filter = GlobalFilter(dim, h=h, w=w)

        # Drop path layer for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Second normalization layer
        self.norm2 = norm_layer(dim)

        # Define hidden dimension for perceptron layer
        mlp_hidden_dim = int(dim * mlp_ratio)

        # Perceptron layer with dropout
        self.mlp = Percep(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Apply normalization and filter, then add residual connection
        x = x + self.drop_path(self.filter(self.norm1(x)))

        # Apply normalization and perceptron, then add residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchyEmbedding(nn.Module):
    """Image to Patch Embedding for breaking down images into patch tokens."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        super(PatchyEmbedding, self).__init__()

        # Store image size and patch size as tuples
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        # Calculating amount of patches
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Convolutional layer to project each patch into the embedding space
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Apply patch embedding and reshape
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class GFNet(nn.Module):
    """Global Filter Network (GFNet) model class with a configurable number of layers and output classes."""

    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2, embed_dim=768, depth=12,
                 mlp_ratio=4., drop_rate=0., drop_path_rate=0., norm_layer=None):
        super(GFNet, self).__init__()

        # Initializes parsed parameters
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Set default normalization layer if none provided
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding layer
        self.patch_embed = PatchyEmbedding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # Calculate number of patches based on embedding dimensions
        num_patches = self.patch_embed.num_patches

        # Positional embedding to encode spatial positions of patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Dropout for positional embeddings
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Set filter dimensions based on patch dimensions
        h = img_size // patch_size
        w = h // 2 + 1

        # Define drop path rates across depth of the model
        dpr = [drop_path_rate for _ in range(depth)]

        # Stack Transformer blocks in sequence
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)
        ])

        # Final normalization layer before classification
        self.norm = norm_layer(embed_dim)

        # Classification head to output predictions
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize positional embedding with truncated normal distribution
        trunc_normal_(self.pos_embed, std=.02)

        # Initialize weights for the network
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Initialize weights for linear layers and normalization layers
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Extract features from input through patch embedding and Transformer blocks
        x = self.patch_embed(x)
        x = x + self.pos_embed  # Add positional embeddings
        x = self.pos_drop(x)  # Apply dropout to the embeddings

        # Pass through each Transformer block
        for blk in self.blocks:
            x = blk(x)

        # Normalize the final output
        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        return x

    def forward(self, x):
        # Forward pass through feature extractor and classification head
        x = self.forward_features(x)
        x = self.head(x)

        return x
