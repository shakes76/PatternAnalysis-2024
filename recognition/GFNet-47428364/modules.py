import math
import torch
import torch.nn as nn
import torch.fft

# Global Filter Block
class GlobalFilterBlock(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.filter = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
        x = x.view(B, a, b, C).to(torch.float32)

        x = torch.fft.rfftn(x, dim=(1, 2), norm="ortho")
        x = x * torch.view_as_complex(self.filter)
        x = torch.fft.irfftn(x, s=(a, b), dim=(1, 2), norm="ortho")

        x = x.reshape(B, N, C)
        return x

# GFNet Block and MLP
class BlockMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., h=14, w=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilterBlock(dim, h=h, w=w)
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
        x = x + self.filter(self.norm1(x)) # Global Filter Block
        x = x + self.mlp(self.norm2(x)) # MLP
        return x

# GFNet Model
class GFNet(nn.Module):
    def __init__(self, embed_dim=384, img_size=224, patch_size=16, in_chans=3, mlp_ratio=4, depth=4, num_classes=1000):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        h = img_size // patch_size
        w = h // 2 + 1

        # Stacked blocks
        self.blocks = nn.Sequential(
            *[BlockMLP(embed_dim, mlp_ratio, h=h, w=w)
            for _ in range(depth)]
        )

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # GFNet blocks
        x = self.blocks(x)

        # Classification head
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)

        x = self.softmax(x)
        return x
