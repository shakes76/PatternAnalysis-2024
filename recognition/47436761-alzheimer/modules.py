from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataset import IMAGE_DIM, PATCH_SIZE, NUM_PATCHES, D_MODEL

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, embed_size=768):
        """Embeds image patches into vectors."""
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_size))

    def forward(self, x):
        # Apply projection to get patches and flatten them
        x = self.projection(x)
        x = x.transpose(1, 2)  # Rearrange to [batch_size, num_patches, embed_size]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        # Concatenate class tokens with the patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=768, num_heads=8, d_mlp=2048, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Self-Attention block with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP block with residual connection
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x


class AlzheimerModel(nn.Module):
    def __init__(self, in_channels, patch_size, embed_size, img_size, num_layers, num_heads, d_mlp, dropout_rate, num_classes=2):
        super(AlzheimerModel, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_size))

        self.encoders = nn.ModuleList([
            TransformerEncoder(d_model=embed_size, num_heads=num_heads, d_mlp=d_mlp, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])

        # MLP Head for classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_size),
            nn.Linear(embed_size, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.positional_embedding

        # Pass through each Transformer Encoder layer
        for encoder in self.encoders:
            x = encoder(x)
        
        # Use the CLS token's output for classification
        output = self.mlp_head(x[:, 0])
        return output