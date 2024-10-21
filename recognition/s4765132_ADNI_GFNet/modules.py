import torch
import torch.nn as nn
import torch.fft

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super(GlobalFilter, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        B, H, W, C = x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x


class GFNet(nn.Module):
    def __init__(self, img_size=224, num_classes=2, channels=3, embed_dim=768, ff_dim=1024, dropout_rate=0.1, num_global_filters=3):
        super(GFNet, self).__init__()

        self.global_filters = nn.ModuleList([GlobalFilter(dim=embed_dim) for _ in range(num_global_filters)])

    
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  
            nn.Linear(ff_dim, embed_dim),
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        
        for global_filter in self.global_filters:
            x = global_filter(x)
    
        x = self.ffn(x)
        x = self.layer_norm(x)
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)
        return x
