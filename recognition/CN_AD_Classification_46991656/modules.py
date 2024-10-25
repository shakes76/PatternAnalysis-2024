# Contains the source code of the GFNet Vision Transformer

import torch
import torch.nn as nn

# Define the TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MLP block with two linear layers
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        x = self.norm2(x + self.mlp(x))
        
        return x

# Define the Vision Transformer Model
class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=32, num_classes=2, dim=64, depth=6, heads=4, mlp_dim=128):
        super(SimpleViT, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2

        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)
        ])

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        patches = x.unfold(2, 32, 32).unfold(3, 32, 32)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, self.num_patches, -1)
        x = self.patch_embedding(patches)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embedding

        for layer in self.transformer:
            x = layer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

# Function to get the Vision Transformer model
def get_vit_model(num_classes=2):
    return SimpleViT(num_classes=num_classes)



