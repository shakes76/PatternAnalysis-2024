import torch
import torch.nn as nn

CONV_EMBEDDING_INPUT_CHANNELS=3
CONV_EMBEDDING_DIM=64
CONV_EMBEDDING_KERNEL_SIZE=7
CONV_EMBEDDING_STRIDE=4
CONV_EMBEDDING_PADDING_SIZE=2
TRANSFORMER_BLOCK_DROPOUT=0.1

CVT_INPUT_CHANNELS=3
CVT_CLASS_COUNT=2
CVT_EMBEDDING_DIM=32
CVT_HEAD_COUNT=8
CVT_HIDDEN_DIM=128
CVT_TRANSFORMER_BLOCK_COUNT=4
CVT_DROPOUT=0.3

class ConvolutionalEmbedding(nn.Module):
    """
    Convolutional Embedding Layer for image patch embedding.

    Args:
        in_channels : Number of input channels.
        embed_dim : Dimensionality of the embedding space.
        kernel_size : Size of the convolutional kernel.
        stride : Stride of the convolution.
        padding : Padding added to the input.
    """
    def __init__(self, in_channels=CONV_EMBEDDING_INPUT_CHANNELS, embed_dim=CONV_EMBEDDING_DIM, kernel_size=CONV_EMBEDDING_KERNEL_SIZE, stride=CONV_EMBEDDING_STRIDE, padding=CONV_EMBEDDING_PADDING_SIZE):
        super(ConvolutionalEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        """
        Forward pass through the convolutional embedding layer.

        Args:
            x : Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embed_dim, new_height, new_width).
        """
        x = self.conv(x)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block consisting of multi-head self-attention and MLP.

    Args:
        embed_dim : Dimensionality of the input and output embeddings.
        num_heads : Number of attention heads.
        mlp_dim : The dimension of the MLP layer.
        dropout : Dropout rate for regularization.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass through the Transformer block.

        Args:
            x : Input tensor of shape (sequence_length, batch_size, embed_dim).

        Returns:
            torch.Tensor: Output tensor of the same shape as input.
        """
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output 
        x = self.norm1(x)
        
        mlp_output = self.mlp(x)
        x = x + mlp_output 
        x = self.norm2(x)
        return x

class CvT(nn.Module):
    """
    Convolutional vision transformer (CvT) model for image classification.

    Args:
        in_channels : Number of input channels.
        num_classes : Number of output classes for classification.
        embed_dim : The dimension of the embedding space.
        num_heads : Number of attention heads in each transformer block.
        mlp_dim : The dimension of the MLP layer in each transformer block.
        num_transformer_blocks : Number of transformer blocks.
        dropout : Dropout rate for regularization.
    """
    def __init__(self, in_channels=CVT_INPUT_CHANNELS, num_classes=CVT_CLASS_COUNT, embed_dim=CVT_EMBEDDING_DIM, num_heads=CVT_HEAD_COUNT, mlp_dim=CVT_HIDDEN_DIM, num_transformer_blocks=CVT_TRANSFORMER_BLOCK_COUNT, dropout=CVT_DROPOUT):
        super(CvT, self).__init__()
        
        self.conv_embedding = ConvolutionalEmbedding(in_channels, embed_dim)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the CvT model.

        Args:
            x : Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.conv_embedding(x)  
        x = x.flatten(2).transpose(1, 2)  
        
        for block in self.transformer_blocks:
            x = block(x)
        
        x = x.mean(dim=1)  
        return self.mlp_head(x)
