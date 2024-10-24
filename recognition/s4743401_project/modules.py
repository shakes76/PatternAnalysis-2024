import torch
import torch.nn as nn
import numpy as np

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=240, patch_size=16, in_channels=1, embed_dim=768):  
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  
        self.embed_dim = embed_dim

        # image into patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  
        x = x.flatten(2)  
        x = x.transpose(1, 2) 
        return x

def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.zeros(seq_len, emb_size)  
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
            nn.ReLU(),  
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
        x = x + self.dropout(self.att(self.ll(x)))  
        x = x + self.dropout(self.ff(self.ll(x)))  
        return x

class VisionTransformer(nn.Module):
    def __init__(self, num_layers, img_size, emb_size, patch_size, num_head, num_class):
        super().__init__()
        self.patchembedding = PatchEmbedding(patch_size=patch_size, img_size=img_size)
        # for non-square images
        #self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)  # Height * Width
        height, width = img_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed = nn.Parameter(PositionEmbedding(self.num_patches + 1, emb_size))  
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))  # Shape: (1, 1, emb_size)
        self.attention = nn.Sequential(*[Block(emb_size, num_head) for _ in range(num_layers)])
        self.ff = nn.Linear(emb_size, num_class)

    def forward(self, x):  
        embeddings = self.patchembedding(x)    
        cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings += self.pos_embed  
        x = self.attention(embeddings) 
        x = self.ff(x[:, 0, :])  
        return x
