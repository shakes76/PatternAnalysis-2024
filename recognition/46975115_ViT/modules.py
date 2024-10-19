import torch
import torch.nn as nn
from collections import OrderedDict

#https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten
class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings using a convolutional projection.
    Each image is divided into patches, and each patch is projected to a vector of hidden dimensions.
    """

    def __init__(self, in_channels: int, hidden_dim: int, patch_size: int):
        """
        Initializes the PatchEmbedding module.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale images).
            hidden_dim (int): Size of the hidden dimension for patch embeddings.
            patch_size (int): Size of each patch.
        """
                
        super().__init__()
        self.conv_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass to generate patch embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Patch embeddings of shape (batch_size, num_patches, hidden_dim).
        """

        # Apply convolution projection        
        x = self.conv_proj(x)
        # Flatten and permute for transformer input
        x = x.flatten(2).permute(0, 2, 1)
        return x

class MLP(nn.Module):
    """
    Multi-layer perceptron block with GELU activation and dropout.
    """

    def __init__(self, mlp: int, hidden: int, dropout: int):
        """
        Initializes the MLP block.

        Args:
            mlp (int): Number of units in the MLP hidden layer.
            hidden (int): Input and output dimensions of the MLP block.
            dropout (int): Dropout probability.
        """

        super().__init__()
        self.fc1 = nn.Linear(hidden, mlp)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the MLP block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the MLP transformations.
        """
                
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block that includes layer normalization, multi-head attention, and MLP.
    """

    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: int, attention_dropout: int):
        """
        Initializes the EncoderBlock.

        Args:
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of hidden embeddings.
            mlp_dim (int): Dimension of the MLP hidden layer.
            dropout (int): Dropout probability.
            attention_dropout (int): Dropout probability for attention weights.
        """
                
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(mlp_dim, hidden_dim, dropout)

    def forward(self, x):
        """
        Forward pass through the encoder block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after the encoder transformations.
        """
                
        # Layer normalization, attention, and residual connection
        x_norm = self.layer_norm_1(x)
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.dropout(attn_output)
        # Apply second layer norm and MLP with residual connection
        mlp_output = self.mlp(self.layer_norm_2(x))
        return x + mlp_output


class Encoder(nn.Module):
    """
    Stacks multiple EncoderBlocks and applies layer normalization at the output.
    """

    def __init__(self, num_layers: int, num_heads: int, hidden_dim: int, mlp_dim: int, dropout: int, attention_dropout: int):
        """
        Initializes the Encoder.

        Args:
            num_layers (int): Number of encoder blocks.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of hidden embeddings.
            mlp_dim (int): Dimension of the MLP hidden layer.
            dropout (int): Dropout probability.
            attention_dropout (int): Dropout probability for attention weights.
        """

        super().__init__()
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)
        self.layers = nn.Sequential(layers)
        self.norm_layer = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Encoded tensor.
        """
                
        x = self.layers(x)
        return self.norm_layer(x)

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model, as described in "An Image is Worth 16x16 Words".
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, num_layers: int = 12, num_heads: int = 12,
                 hidden_dim: int = 768, mlp_dim: int = 3072, num_classes: int = 2, in_channels: int = 1,
                 attention_dropout: float = 0., dropout: float = 0.):
        """
        Initializes the Vision Transformer (ViT).

        Args:
            image_size (int): Size of input images.
            patch_size (int): Size of patches.
            num_layers (int): Number of encoder layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of hidden embeddings.
            mlp_dim (int): Dimension of MLP hidden layer.
            num_classes (int): Number of output classes.
            in_channels (int): Number of input channels.
            attention_dropout (float): Dropout probability for attention.
            dropout (float): Dropout probability.
        """
                
        super().__init__()
        # Patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels, hidden_dim, patch_size)
        self.n_patches = (image_size // patch_size) ** 2

        # Class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.n_patches += 1  # Add one for the class token

        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.empty(1, self.n_patches, hidden_dim).normal_(std=0.02))

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        self.transformer_encoder = Encoder(num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout)

        # Classification head
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits for each class.
        """
                
        batch_size = x.shape[0]
        # Apply patch embedding
        x = self.patch_embedding(x)

        # Add class token to the input
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)

        # Add positional embedding
        x = x + self.positional_embedding
        x = self.dropout(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Apply classifier to the class token and return
        return self.classifier_head(x[:, 0])


