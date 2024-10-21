# modules.py

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: (batch_size, in_channels, img_size, img_size)
        x = self.proj(x)  # (batch_size, emb_size, num_patches^(1/2), num_patches^(1/2))
        x = x.flatten(2)  # (batch_size, emb_size, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, emb_size)
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=12, dropout=0.):
        super(MultiHeadSelfAttention, self).__init__()
        assert emb_size % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc_out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_tokens, emb_size = x.size()

        qkv = self.qkv(x)  # (batch_size, num_tokens, 3 * emb_size)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, num_tokens, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]  # Each: (batch_size, num_heads, num_tokens, head_dim)

        # Scaled Dot-Product Attention
        energy = torch.matmul(queries, keys.transpose(-2, -1))  # (batch_size, num_heads, num_tokens, num_tokens)
        scaling = float(self.head_dim) ** -0.5
        energy = energy * scaling
        attention = torch.softmax(energy, dim=-1)  # (batch_size, num_heads, num_tokens, num_tokens)
        attention = self.dropout(attention)

        out = torch.matmul(attention, values)  # (batch_size, num_heads, num_tokens, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, emb_size)  # (batch_size, num_tokens, emb_size)

        out = self.fc_out(out)  # (batch_size, num_tokens, emb_size)
        out = self.dropout(out)
        return out
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=12, ff_dim=3072, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.msa = MultiHeadSelfAttention(emb_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-Head Self-Attention with residual connection
        x = x + self.msa(self.norm1(x))

        # Feed-Forward Network with residual connection
        x = x + self.ffn(self.norm2(x))
        return x
