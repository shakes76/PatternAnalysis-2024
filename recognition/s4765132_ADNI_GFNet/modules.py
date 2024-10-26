import torch
import torch.nn as nn
import torch.fft

class GlobalFilter(nn.Module):
    def __init__(self, dim):
        super(GlobalFilter, self).__init__()
        self.dim = dim

    def forward(self, x):
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        # Dynamically generate complex_weight to match the transformed x dimensions
        _, new_H, new_W, _ = x.shape
        complex_weight = torch.randn(new_H, new_W, self.dim, dtype=torch.complex64, device=x.device) * 0.02

        # Apply frequency domain filtering
        x = x * complex_weight
        # Convert back to spatial domain using inverse FFT with the original height and width
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x


class GFNet(nn.Module):
    def __init__(self, img_size=224, num_classes=2, in_chans=3, embed_dim=768, ff_dim=1024, dropout_rate=0.1, num_global_filters=3):
        super(GFNet, self).__init__()

        # Convolutional layer to project input images to the embedding dimension
        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        # Module with a series of global filters
        self.global_filters = nn.ModuleList([GlobalFilter(dim=embed_dim) for _ in range(num_global_filters)])

        # Feed-forward network layers
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(ff_dim, embed_dim),
        )

        # Normalization layer and classification layer
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Project input to desired embedding dimension using convolution
        x = self.conv1(x)  
        B, C, H, W = x.shape

        # Permute tensor from [B, C, H, W] to [B, H, W, C] to match GlobalFilter
        x = x.permute(0, 2, 3, 1)

        for global_filter in self.global_filters:
            x = global_filter(x)

        # Permute back to [B, C, H, W] after filtering
        x = x.permute(0, 3, 1, 2)  
        x = self.pool(x) 
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

