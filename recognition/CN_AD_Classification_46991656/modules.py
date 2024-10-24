# Contains the source code of the GFNet Vision Transformer

import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)  # (batch_size, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)  # Self-attention
        x = self.norm1(x + attn_out)  # Residual connection + normalization
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)  # Another residual connection + normalization
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, patch_size=16, embed_dim=192, num_heads=3, ff_hidden_dim=768, num_layers=12):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)
        for block in self.transformer_blocks:
            x = block(x)  # Process through each transformer block
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)  # Classification head
        return x

def get_vit_model(num_classes=2):
    return VisionTransformer(num_classes=num_classes)
    

