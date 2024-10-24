import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=240, patch_size=16, in_channels=1, embed_dim=768):  # Adjust in_channels
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # Height * Width
        self.embed_dim = embed_dim

        # Projection of image into patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x

def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.zeros(seq_len, emb_size)  # Initialize with zeros
    for i in range(seq_len):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000, j / emb_size))) if j % 2 == 0 else np.cos(i / (pow(10000, (j - 1) / emb_size)))
    return embeddings  # No need to convert to tensor here, return is already torch

class MultiHead(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super(MultiHead, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        return self.fc(attn_output)

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),  # Add activation function
            nn.Linear(4 * emb_size, emb_size)
        )

    def forward(self, x):
        return self.ff(x)

class Block(nn.Module):
    def __init__(self, emb_size, num_head):
        super().__init__()
        self.att = MultiHead(emb_size, num_head)
        self.ll = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)
        self.ff = FeedForward(emb_size)

    def forward(self, x):
        x = x + self.dropout(self.att(self.ll(x)))  # Residual connection
        x = x + self.dropout(self.ff(self.ll(x)))  # Residual connection
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_layers, img_size, emb_size, patch_size, num_head, num_class):
        super().__init__()
        self.patchemb = PatchEmbedding(patch_size=patch_size, img_size=img_size)
        # Correct calculation for non-square images
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # Height * Width
        self.pos_embed = nn.Parameter(PositionEmbedding(self.num_patches + 1, emb_size))  # +1 for class token
        self.attention = nn.Sequential(*[Block(emb_size, num_head) for _ in range(num_layers)])
        self.ff = nn.Linear(emb_size, num_class)

    def forward(self, x):  # x -> (b, c, h, w)
        embeddings = self.patchemb(x)    
        embeddings += self.pos_embed  # Add positional embedding
        x = self.attention(embeddings)  # (b, num_patches, emb_dim)
        x = self.ff(x[:, 0, :])  # Use the class token for final output
        return x
