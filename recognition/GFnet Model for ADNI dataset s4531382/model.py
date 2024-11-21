import torch
import torch.nn as nn
import torch.fft
import math
from functools import partial
from collections import OrderedDict
from timm.layers import DropPath, to_2tuple, trunc_normal_

class FeedForwardNetwork(nn.Module):
    """
    A simple Feed-Forward Neural Network module consisting of two linear layers
    with an activation function and dropout in between.

    Args:
        input_dim (int): Dimension of the input features.
        hidden_dim (int, optional): Dimension of the hidden layer. Defaults to input_dim.
        output_dim (int, optional): Dimension of the output layer. Defaults to input_dim.
        activation (nn.Module, optional): Activation function class. Defaults to nn.GELU.
        dropout (float, optional): Dropout rate. Defaults to 0.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation=nn.GELU, dropout=0):
        super(FeedForwardNetwork, self).__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        self.layer1 = nn.Linear(input_dim, hidden_dim)  # First linear transformation
        self.act = activation()                         # Activation function
        self.layer2 = nn.Linear(hidden_dim, output_dim) # Second linear transformation
        self.drop = nn.Dropout(dropout)                 # Dropout layer

    def forward(self, x):
        """
        Forward pass of the FeedForwardNetwork.

        Args:
            x (Tensor): Input tensor of shape (batch_size, ..., input_dim).

        Returns:
            Tensor: Output tensor after applying two linear layers, activation, and dropout.
        """
        x = self.layer1(x)   # Apply first linear layer
        x = self.act(x)      # Apply activation function
        x = self.drop(x)     # Apply dropout
        x = self.layer2(x)   # Apply second linear layer
        x = self.drop(x)     # Apply dropout again
        return x

class SpectralFilter(nn.Module):
    """
    A spectral filtering module that applies learnable frequency-domain filters
    to the input using Fourier transforms.

    Args:
        channels (int): Number of input channels.
        height (int, optional): Height dimension for reshaping. Defaults to 14.
        width (int, optional): Width dimension for reshaping. Defaults to 8.
    """
    def __init__(self, channels, height=14, width=8):
        super().__init__()
        # Initialize filter weights as learnable parameters with small random values
        self.filter_weights = nn.Parameter(torch.randn(height, width, channels, 2, dtype=torch.float32) * 0.02)
        self.height = height
        self.width = width

    def forward(self, x, spatial_dims=None):
        """
        Forward pass of the SpectralFilter.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, channels).
            spatial_dims (tuple, optional): Tuple of (height, width) to reshape tokens. 
                                            If None, assumes square spatial dimensions.

        Returns:
            Tensor: Filtered tensor of shape (batch_size, num_tokens, channels).
        """
        batch_size, num_tokens, channels = x.shape
        if spatial_dims is None:
            height = width = int(math.sqrt(num_tokens))  # Assume square if spatial_dims not provided
        else:
            height, width = spatial_dims

        x = x.view(batch_size, height, width, channels)  # Reshape to spatial dimensions
        x = x.to(torch.float32)                           # Ensure tensor is in float32

        # Apply 2D real FFT to transform to frequency domain
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        filters = torch.view_as_complex(self.filter_weights)  # Convert filter weights to complex numbers
        x = x * filters                                       # Apply spectral filters
        # Apply inverse 2D real FFT to transform back to spatial domain
        x = torch.fft.irfft2(x, s=(height, width), dim=(1, 2), norm='ortho')

        x = x.reshape(batch_size, num_tokens, channels)  # Reshape back to original token format
        return x

class TransformerBlock(nn.Module):
    """
    A Transformer block that includes normalization, spectral filtering, 
    drop path, and a feed-forward network.

    Args:
        channels (int): Number of input channels.
        mlp_ratio (float, optional): Ratio to determine hidden dimension in MLP. Defaults to 4.0.
        dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        drop_path (float, optional): Drop path rate. Defaults to 0.0.
        activation (nn.Module, optional): Activation function class. Defaults to nn.GELU.
        normalization (nn.Module, optional): Normalization layer class. Defaults to nn.LayerNorm.
        height (int, optional): Height dimension for spectral filtering. Defaults to 14.
        width (int, optional): Width dimension for spectral filtering. Defaults to 8.
    """
    def __init__(self, channels, mlp_ratio=4.0, dropout=0.0, drop_path=0.0,
                 activation=nn.GELU, normalization=nn.LayerNorm, height=14, width=8):
        super(TransformerBlock, self).__init__()
        self.norm1 = normalization(channels)  # First normalization layer
        self.filter = SpectralFilter(channels, height=height, width=width)  # Spectral filter
        # DropPath layer for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = normalization(channels)  # Second normalization layer
        FFN_hidden_dim = int(channels * mlp_ratio)  # Compute hidden dimension for MLP
        self.mlp = FeedForwardNetwork(input_dim=channels, hidden_dim=FFN_hidden_dim, activation=activation, dropout=dropout)  # Feed-forward network

    def forward(self, x):
        """
        Forward pass of the TransformerBlock.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_tokens, channels).

        Returns:
            Tensor: Output tensor after applying Transformer operations.
        """
        # Apply normalization, spectral filtering, normalization, MLP, drop path, and add residual
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class PatchEmbedding(nn.Module):
    """
    A module to embed image patches into a sequence of feature vectors.

    Args:
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of each patch. Defaults to 16.
        in_channels (int, optional): Number of input channels. Defaults to 3.
        embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = to_2tuple(image_size)  # Ensure image_size is a tuple
        self.patch_size = to_2tuple(patch_size)  # Ensure patch_size is a tuple
        # Calculate the number of patches
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])

        # Convolutional layer to project image patches to embedding dimension
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        """
        Forward pass of the PatchEmbedding module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Raises:
            ValueError: If input tensor does not have 4 dimensions.

        Returns:
            Tensor: Embedded patches of shape (batch_size, num_patches, embed_dim).
        """
        if len(x.shape) != 4:
            raise ValueError(f"Expected input of shape (batch_size, channels, height, width), but got {x.shape}")
        batch_size, channels, height, width = x.shape
        # Apply convolution to get patch embeddings, then flatten and transpose
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class GlobalFilterNetwork(nn.Module):
    """
    A global filter-based neural network for image classification, integrating patch embedding,
    positional encoding, multiple Transformer blocks with spectral filtering, and a classifier.

    Args:
        image_size (int, optional): Size of the input image. Defaults to 224.
        patch_size (int, optional): Size of each patch. Defaults to 16.
        in_channels (int, optional): Number of input channels. Defaults to 3.
        num_classes (int, optional): Number of output classes. Defaults to 1.
        embed_dim (int, optional): Dimension of the embedding. Defaults to 768.
        depth (int, optional): Number of Transformer blocks. Defaults to 12.
        mlp_ratio (float, optional): Ratio to determine hidden dimension in MLP. Defaults to 4.0.
        representation_size (int, optional): Size of the representation before classification. Defaults to None.
        uniform_drop (bool, optional): Whether to use uniform drop path rates. Defaults to False.
        dropout (float, optional): Dropout rate for positional embedding and MLP. Defaults to 0.0.
        drop_path_rate (float, optional): Maximum drop path rate. Defaults to 0.0.
        normalization (callable, optional): Normalization layer constructor. Defaults to nn.LayerNorm.
        classifier_dropout (float, optional): Dropout rate before the classifier. Defaults to 0.0.
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1, embed_dim=768, depth=12,
                 mlp_ratio=4.0, representation_size=None, uniform_drop=False,
                 dropout=0.0, drop_path_rate=0.0, normalization=None, classifier_dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        # Set default normalization if not provided
        normalization = normalization or partial(nn.LayerNorm, eps=1e-6)

        # Initialize patch embedding module
        self.patch_embed = PatchEmbedding(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable positional embeddings
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.positional_dropout = nn.Dropout(p=dropout)  # Dropout after adding positional embeddings

        height = image_size // patch_size
        width = height // 2 + 1  # Define width based on height

        # Configure drop path rates for each Transformer block
        if uniform_drop:
            print(f'Using uniform drop path with rate {drop_path_rate}')
            drop_rates = [drop_path_rate for _ in range(depth)]
        else:
            print(f'Using linear drop path with max rate {drop_path_rate}')
            drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Initialize a list of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                channels=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout, drop_path=drop_rates[i],
                normalization=normalization, height=height, width=width)
            for i in range(depth)
        ])

        self.norm = normalization(embed_dim)  # Final normalization layer

        # Optional pre-logits layer for representation learning
        if representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
            self.embed_dim = representation_size
        else:
            self.pre_logits = nn.Identity()

        # Classification layer
        self.classifier = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Optional dropout before the classifier
        if classifier_dropout > 0.0:
            print(f'Applying classifier dropout of {classifier_dropout}')
            self.last_dropout = nn.Dropout(p=classifier_dropout)
        else:
            self.last_dropout = nn.Identity()

        # Initialize positional embeddings with truncated normal distribution
        trunc_normal_(self.positional_embedding, std=0.02)
        self.apply(self.initialize_weights)  # Initialize weights of all modules

    def initialize_weights(self, module):
        """
        Initialize weights of the model modules.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)  # Initialize weights with truncated normal
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)   # Initialize biases to zero
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)       # Initialize LayerNorm biases to zero
            nn.init.constant_(module.weight, 1.0)     # Initialize LayerNorm weights to one

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Specify parameters that should not have weight decay applied.

        Returns:
            set: Set of parameter names without weight decay.
        """
        return {'positional_embedding'}

    def forward_features(self, x):
        """
        Extract features from the input by passing through patch embedding,
        positional encoding, Transformer blocks, and normalization.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Feature tensor of shape (batch_size, embed_dim).
        """
        batch_size = x.shape[0]
        x = self.patch_embed(x)                # Embed image patches
        x = x + self.positional_embedding      # Add positional embeddings
        x = self.positional_dropout(x)         # Apply dropout to positional embeddings

        for block in self.transformer_blocks:
            x = block(x)                        # Pass through each Transformer block

        x = self.norm(x).mean(dim=1)           # Apply final normalization and average pooling
        return x

    def forward(self, x):
        """
        Forward pass of the GlobalFilterNetwork.

        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.forward_features(x)            # Extract features
        x = self.last_dropout(x)                # Apply dropout before classifier
        x = self.classifier(x)                  # Classify
        x = x.squeeze()                         # Squeeze the output tensor
        return x
