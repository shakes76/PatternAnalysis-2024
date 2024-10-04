import torch
import torch.nn as nn
import torch.fft

# Global Filter Block
class GlobalFilterBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.filter = nn.Parameter(torch.randn(1, dim, 1, 1)) # Learnable Frequency Filter
    
    def forward(self, x):
        _, _, H, W = x.shape

        x = torch.fft.rfftn(x, dim=(-2, -1), norm="ortho") # Fast Fourier Transform
        x = x * self.filter # Apply filter
        x = torch.fft.irfftn(x, s=(H, W), dim=(-2, -1), norm="ortho") # Inverse Fast Fourier Transform

        return x

# GFNet block and MLP
class BlockMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.global_filter = GlobalFilterBlock(dim)
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        x = x + self.global_filter(self.norm1(x)) # Global Filter
        x = x + self.mlp(self.norm2(x)) # MLP
        return x

# GFNet Model
class GFNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=2, embed_dim=768, depth=12, mlp_ratio=4.):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Stacked blocks
        self.blocks = nn.Sequential(
            *[BlockMLP(embed_dim, mlp_ratio) for _ in range(depth)]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        # GfNet Blocks
        x = self.blocks(x)
        # Classification Head
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        return x
