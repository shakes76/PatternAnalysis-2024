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
