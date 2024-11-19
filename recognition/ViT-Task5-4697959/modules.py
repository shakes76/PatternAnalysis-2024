import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # A convolution layer is used to convert each image patch into an embedding vector.
        # `proj` downsamples the input image from its original size to a smaller grid of patches.
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        # Calculate the total number of patches in the image based on the image and patch sizes.
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        # Apply the convolution to get a smaller grid of patches.
        x = self.proj(x)  # Output shape: (batch_size, emb_size, num_patches^(1/2), num_patches^(1/2))
        # Flatten the grid of patches into a single sequence.
        x = x.flatten(2)  # Output shape: (batch_size, emb_size, num_patches)
        # Transpose to get the shape needed for the transformer (batch_size, num_patches, emb_size).
        x = x.transpose(1, 2)  # Output shape: (batch_size, num_patches, emb_size)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=12, dropout=0.):
        super(MultiHeadSelfAttention, self).__init__()
        # Ensure the embedding size is divisible by the number of heads.
        assert emb_size % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.emb_size = emb_size
        self.num_heads = num_heads
        # Calculate the dimension for each attention head.
        self.head_dim = emb_size // num_heads

        # Linear layer to compute queries, keys, and values in a single matrix operation.
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, emb_size = x.size()

        # Generate the queries, keys, and values from the input `x`.
        qkv = self.qkv(x)  # Output shape: (batch_size, num_tokens, 3 * emb_size)
        # Reshape the combined tensor into three separate parts: queries, keys, and values.
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        # Permute to separate queries, keys, and values for each attention head.
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: (3, batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, num_tokens, head_dim)

        # Compute attention scores (energy) using scaled dot-product.
        energy = torch.matmul(queries, keys.transpose(-2, -1))  # Shape: (batch_size, num_heads, num_tokens, num_tokens)
        scaling = float(self.head_dim) ** -0.5
        energy = energy * scaling
        # Apply softmax to get attention weights.
        attention = torch.softmax(energy, dim=-1) 
        # Apply dropout to the attention weights.
        attention = self.dropout(attention)

        # Compute the final output by applying attention weights to the values.
        out = torch.matmul(attention, values)  # Shape: (batch_size, num_heads, num_tokens, head_dim)
        # Concatenate the heads and flatten them back into the original embedding size.
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, emb_size)  # Shape: (batch_size, num_tokens, emb_size)

        # Apply a final linear transformation and dropout.
        out = self.fc_out(out)  # Shape: (batch_size, num_tokens, emb_size)
        out = self.dropout(out)
        return out
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        # Layer normalization before Multi-Head Self-Attention.
        self.norm1 = nn.LayerNorm(emb_size)
        # Multi-Head Self-Attention module.
        self.msa = MultiHeadSelfAttention(emb_size, num_heads, dropout)
        # Layer normalization before Feed-Forward Network.
        self.norm2 = nn.LayerNorm(emb_size)
        # Feed-Forward Network with dropout for regularization.
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Self-Attention with residual connection.
        x = x + self.msa(self.norm1(x))

        # Feed-Forward Network with residual connection.
        x = x + self.ffn(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_size=768, num_heads=12, depth=12, ff_dim=3072, num_classes=2, dropout=0.1, cls_token=True):
        super(VisionTransformer, self).__init__()
        # Patch embedding layer to convert the input image into a sequence of patch embeddings.
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        num_patches = self.patch_embed.num_patches

        # Classification token (optional) to summarize the image.
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size)) if cls_token else None

        # Positional embedding to give each patch a unique position in the sequence.
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + (1 if cls_token else 0), emb_size))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder consisting of multiple layers.
        self.encoder = nn.Sequential(
            *[TransformerEncoderBlock(emb_size, num_heads, ff_dim, dropout) for _ in range(depth)]
        )

        # Normalization layer before classification.
        self.norm = nn.LayerNorm(emb_size)
        # Classification head that outputs class probabilities.
        self.head = nn.Linear(emb_size, num_classes)

        # Initialize model weights.
        self._init_weights()

    def _init_weights(self):
        # Initialize the positional embedding and other parameters.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # Convert the input image into patch embeddings.
        x = self.patch_embed(x)  # Shape: (batch_size, num_patches, emb_size)

        if self.cls_token is not None:
            # Add the classification token to the sequence.
            batch_size = x.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, emb_size)
            x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, emb_size)

        # Add positional embeddings to the input.
        x = x + self.pos_embed  
        x = self.dropout(x)

        # Pass through the transformer encoder.
        x = self.encoder(x)

        # Apply normalization before classification.
        x = self.norm(x)

        # If a classification token is used, extract it. Otherwise, use average pooling.
        if self.cls_token is not None:
            x = x[:, 0]  # Extract the classification token.
        else:
            x = x.mean(dim=1)  # Global average pooling.

        # Pass through the classification head to get final predictions.
        x = self.head(x)
        return x
