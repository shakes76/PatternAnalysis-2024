# Modified code from https://github.com/raoyongming/GFNet/blob/master/gfnet.py
import math
import torch
import torch.nn as nn
import torch.fft

# GFNet Global Filter Class
class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        x = x.reshape(B, N, C)
        return x

# GFNet MLP Class
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# GFNet Block Class
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., h=14, w=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim, drop)

    def forward(self, x):
        x = x + self.mlp(self.norm2(self.filter(self.norm1(x))))
        return x

# GFNet Model
class GFNet(nn.Module):
    def __init__(self, embed_dim=768, img_size=224, patch_size=16, in_chans=3, mlp_ratio=4, depth=12, drop_rate=0.1, num_classes=1000):
        super().__init__()

        # Patch embedding
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        # Stacked blocks
        self.blocks = nn.ModuleList([
            Block(
                embed_dim, mlp_ratio, drop_rate, h=h, w=w)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(1)

        # Classification head
        x = self.head(x)

        return x
