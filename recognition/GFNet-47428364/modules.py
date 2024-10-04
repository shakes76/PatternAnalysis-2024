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

