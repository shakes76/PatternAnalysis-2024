"""
This script provides a modified version of GFNet, originally implemented by Yongming Rao [https://github.com/raoyongming/GFNet].

Modifications have been made to adapt the model for binary classification tasks, ensuring compatibility adjustments where necessary.
"""

import math
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft

class FeedForwardNetwork(nn.Module):
    """Multilayer perceptron (MLP) with two linear layers and an activation function.

    The MLP serves as a feed-forward network within the transformer block. It helps the model learn complex
    transformations of the input features, enabling it to capture non-linear patterns in the data.

    Args:
        input_dim (int): Size of each input sample.
        hidden_dim (int, optional): Size of the hidden layer. Defaults to input_dim.
        output_dim (int, optional): Size of each output sample. Defaults to input_dim.
        activation_layer (nn.Module, optional): Activation function to use. Defaults to nn.GELU.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, activation_layer=nn.GELU, dropout_rate=0.):
        super().__init__()
        # Set hidden and output feature sizes
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        # First linear layer transforms input features to hidden features
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Activation function introduces non-linearity
        self.activation = activation_layer()
        # Second linear layer transforms hidden features to output features
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Dropout layers help prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)           # Linear transformation
        x = self.activation(x)    # Apply activation function
        x = self.dropout(x)       # Apply dropout
        x = self.fc2(x)           # Another linear transformation
        x = self.dropout(x)       # Apply dropout
        return x

class GlobalFrequencyFilter(nn.Module):
    """Applies a global filter in the frequency domain.

    By performing operations in the frequency domain, this layer captures global patterns in the data,
    which might be challenging to capture using only spatial domain operations. This enhances the model's
    ability to understand long-range dependencies and global context within images.

    Args:
        embed_dim (int): Embedding dimension of the input features.
        height (int): Height dimension after reshaping the input for FFT.
        width (int): Width dimension after reshaping the input for FFT.
    """
    def __init__(self, embed_dim, height=14, width=8):
        super().__init__()
        # Initialize complex weights for the frequency filter with small random values
        self.complex_weight = nn.Parameter(torch.randn(height, width, embed_dim, 2, dtype=torch.float32) * 0.02)
        self.width = width
        self.height = height

    def forward(self, x, spatial_size=None):
        # x shape: (batch_size, num_patches, embed_dim)
        batch_size, num_patches, channels = x.shape
        if spatial_size is None:
            # Calculate height and width if not provided
            height = width = int(math.sqrt(num_patches))
        else:
            height, width = spatial_size

        # Reshape x to (batch_size, height, width, channels)
        x = x.view(batch_size, height, width, channels)
        x = x.to(torch.float32)  # Ensure data type is float32 for FFT operations

        # Perform 2D real FFT to convert spatial data to frequency domain
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        # Convert real and imaginary parts to complex numbers
        weight = torch.view_as_complex(self.complex_weight)

        # Element-wise multiplication in the frequency domain (applying the learned filter)
        x = x * weight

        # Perform inverse FFT to convert back to spatial domain
        x = torch.fft.irfft2(x, s=(height, width), dim=(1, 2), norm='ortho')

        # Reshape back to original shape (batch_size, num_patches, channels)
        x = x.reshape(batch_size, num_patches, channels)

        return x

class GFNetBlock(nn.Module):
    """Transformer block with a global frequency filter and MLP.

    Each block helps the model to learn representations that capture both global and local features.
    The global filter captures global dependencies, while the MLP captures complex patterns through non-linear transformations.

    Args:
        embed_dim (int): Embedding dimension of the input features.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.
        drop_path_rate (float, optional): Drop path rate for stochastic depth. Defaults to 0.
        activation_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        height (int, optional): Height for the global filter. Defaults to 14.
        width (int, optional): Width for the global filter. Defaults to 8.
    """
    def __init__(self, embed_dim, mlp_ratio=4., dropout_rate=0., drop_path_rate=0., activation_layer=nn.GELU, norm_layer=nn.LayerNorm, height=14, width=8):
        super().__init__()
        # Layer normalization helps stabilize training and improves convergence
        self.norm1 = norm_layer(embed_dim)
        # Global frequency filter captures global context in frequency domain
        self.global_filter = GlobalFrequencyFilter(embed_dim, height=height, width=width)
        # Drop path helps regularize the network and prevent overfitting
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # Second normalization layer
        self.norm2 = norm_layer(embed_dim)
        # MLP layer for non-linear transformations
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.feed_forward = FeedForwardNetwork(input_dim=embed_dim, hidden_dim=mlp_hidden_dim, activation_layer=activation_layer, dropout_rate=dropout_rate)

    def forward(self, x):
        # Apply normalization and global filter, then add residual connection
        x = x + self.drop_path(self.global_filter(self.norm1(x)))
        # Apply normalization and MLP, then add residual connection
        x = x + self.drop_path(self.feed_forward(self.norm2(x)))
        return x

class PatchEmbedding(nn.Module):
    """Converts an image into a sequence of patch embeddings.

    Splitting the image into patches allows the model to process images as sequences, similar to words in a sentence.
    Each patch is projected into an embedding space, enabling the model to learn representations for image patches.

    Args:
        img_size (int or tuple): Input image size.
        patch_size (int or tuple): Size of each patch.
        in_channels (int, optional): Number of input channels (e.g., 3 for RGB). Defaults to 3.
        embed_dim (int, optional): Embedding dimension for each patch. Defaults to 768.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        # Ensure image and patch sizes are tuples
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # Calculate the number of patches
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Convolutional layer to project patches into the embedding space
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        B, C, H, W = x.shape
        # Check if input image size matches expected size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match expected size ({self.img_size[0]}*{self.img_size[1]})."
        # Apply convolution to extract patch embeddings
        x = self.projection(x).flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x

class DownsampleLayer(nn.Module):
    """Downsamples the input feature map.

    Used to reduce the spatial dimensions of the feature map, enabling the model to capture hierarchical features
    and reducing computational complexity.

    Args:
        img_size (int, optional): Input image size. Defaults to 56.
        input_dim (int, optional): Input feature dimension. Defaults to 64.
        output_dim (int, optional): Output feature dimension after downsampling. Defaults to 128.
    """
    def __init__(self, img_size=56, input_dim=64, output_dim=128):
        super().__init__()
        self.img_size = img_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Convolutional layer for downsampling
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)
        self.num_patches = (img_size // 2) * (img_size // 2)

    def forward(self, x):
        # x shape: (batch_size, num_patches, input_dim)
        B, N, C = x.size()
        # Reshape and permute to match convolutional layer input requirements
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)  # Shape: (batch_size, input_dim, height, width)
        # Apply downsampling convolution
        x = self.projection(x).permute(0, 2, 3, 1)  # Shape: (batch_size, new_height, new_width, output_dim)
        # Flatten back to sequence format
        x = x.reshape(B, -1, self.output_dim)  # Shape: (batch_size, new_num_patches, output_dim)
        return x

class GFNet(nn.Module):
    """
    Modified GFNet model for binary classification tasks.

    GFNet leverages global filtering in the frequency domain within a transformer architecture.
    By adapting it for binary classification, we adjust the final layers and ensure compatibility
    with tasks that require distinguishing between two classes.

    Parameters:
        img_size (int or tuple): Input image dimensions.
        patch_size (int or tuple): Size of each image patch.
        in_channels (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        depth (int): Number of transformer blocks.
        mlp_ratio (float): Ratio of hidden layer size to embedding dimension in MLP.
        representation_size (Optional[int]): Size of the representation layer before the classifier.
        uniform_drop (bool): Whether to use uniform drop path rates.
        dropout_rate (float): Dropout rate after positional embedding.
        drop_path_rate (float): Rate for stochastic depth (DropPath).
        norm_layer (nn.Module): Normalization layer.
        classifier_dropout (float): Dropout rate before the classifier.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 dropout_rate=0., drop_path_rate=0., norm_layer=None,
                 classifier_dropout=0):
        super().__init__()
        num_classes = 1  # Output dimension is 1 for binary classification
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # Embedding dimension
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch embedding layer converts images to patch embeddings
        self.patch_embedding = PatchEmbedding(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embedding.num_patches

        # Learnable positional embeddings to retain spatial information
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        # Dropout after adding positional embeddings
        self.positional_dropout = nn.Dropout(p=dropout_rate)

        # Dimensions for the global filter
        height = img_size // patch_size
        width = height // 2 + 1  # For real FFT, output width is n//2 + 1

        # Define stochastic depth rates for each block
        if uniform_drop:
            print('Applying uniform drop path with expected rate', drop_path_rate)
            drop_path_rates = [drop_path_rate for _ in range(depth)]
        else:
            print('Applying linear drop path with expected rate', drop_path_rate * 0.5)
            drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Create transformer blocks
        self.blocks = nn.ModuleList([
            GFNetBlock(
                embed_dim=embed_dim, mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate, drop_path_rate=drop_path_rates[i],
                norm_layer=norm_layer, height=height, width=width)
            for i in range(depth)])

        # Final normalization layer
        self.norm = norm_layer(embed_dim)

        # Optional representation layer before the classifier
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head for binary classification
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Optional dropout before the classifier to prevent overfitting
        if classifier_dropout > 0:
            print('Applying dropout of %.2f before the classifier' % classifier_dropout)
            self.classifier_dropout = nn.Dropout(p=classifier_dropout)
        else:
            self.classifier_dropout = nn.Identity()

        # Initialize positional embeddings
        trunc_normal_(self.positional_embedding, std=.02)
        # Initialize weights of the model
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """Initialize model weights.

        Proper initialization is crucial for training deep networks to ensure stable gradients
        and convergence.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            # Initialize linear layers with truncated normal distribution
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to zero
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)     # Initialize LayerNorm biases to zero
            nn.init.constant_(m.weight, 1.0) # Initialize LayerNorm weights to one

    @torch.jit.ignore
    def no_weight_decay(self):
        # Specify parameters that should not be regularized with weight decay
        return {'positional_embedding', 'cls_token'}

    def get_classifier(self):
        # Return the classifier head
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        # Reset the classifier with a new number of classes
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        Extract features from the input images.

        The feature extraction involves converting images to patch embeddings,
        adding positional embeddings, and passing through transformer blocks.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Feature representation of the input.
        """
        batch_size = x.shape[0]
        # Convert images to patch embeddings
        x = self.patch_embedding(x)  # Shape: (batch_size, num_patches, embed_dim)
        # Add positional embeddings
        x = x + self.positional_embedding
        # Apply dropout
        x = self.positional_dropout(x)

        # Pass through each transformer block
        for blk in self.blocks:
            x = blk(x)

        # Apply final normalization
        x = self.norm(x)
        # Aggregate features by taking the mean across patches
        x = x.mean(1)  # Shape: (batch_size, embed_dim)
        return x

    def forward(self, x):
        """
        Forward pass of the model.

        Processes the input through feature extraction and classification layers to produce
        logits for binary classification.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits for binary classification.
        """
        # Extract features from the input
        x = self.forward_features(x)
        # Apply dropout before classification
        x = self.classifier_dropout(x)
        # Pass through the classifier head
        x = self.head(x)
        # Remove unnecessary dimensions
        x = x.squeeze()
        return x
