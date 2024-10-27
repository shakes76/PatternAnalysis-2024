import torch
import torch.nn as nn
import numpy as np
import torch.fft

class PatchyEmbedding(nn.Module):
    def __init__(self, image_size=224, patch_size=16, latent_size=768, batch_size=32):
        super(PatchyEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.latent_size = latent_size
        self.batch_size = batch_size

        # Downsample the images (project patches into embedded space)
        # More efficient than image worth more 16x16 paper??
        self.project = nn.Conv2d(3, self.latent_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        x = self.project(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim, num_patches, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(num_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2) * (-np.log(10000.0) / latent_dim))
        pe = torch.zeros(1, num_patches, latent_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class GlobalFilterLayer(nn.Module):
    def __init__(self, embed_dim):
        super(GlobalFilterLayer, self).__init__()
        self.global_filter = nn.Parameter(torch.randn(1, embed_dim, 1, 1, dtype=torch.cfloat))

    def forward(self, x):
        x = x.transpose(1, 2).unsqueeze(-1)  # Reshape to (batch_size, embed_dim, num_patches, 1)

        # Apply FFt
        freq_x = torch.fft.fft2(x, dim=(-3, -2))
        filtered_freq_x = freq_x * self.global_filter

        # Inv FFt
        output = torch.fft.ifft2(filtered_freq_x, dim=(-3, -2))
        output = output.squeeze(-1).transpose(1, 2)

        return output.real


class GFNetBlock(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.1):
        super(GFNetBlock, self).__init__()
        self.global_filter = GlobalFilterLayer(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        filtered_x = self.global_filter(self.norm1(x))
        x = x + self.dropout(filtered_x)

        mlp_output = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_output)
        return x


class GFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, mlp_dim=3072):
        super(GFNet, self).__init__()
        # Patch embedding layer (same as ViT)
        self.patch_embed = PatchyEmbedding(image_size=img_size, patch_size=patch_size, latent_size=embed_dim)

        # Positional encoding (same as ViT)
        num_patches = (img_size // patch_size) ** 2
        #self.pos_encoding = PositionalEncoding(embed_dim, num_patches)
        self.pos_encoding = PositionalEncoding(embed_dim, num_patches + 1)

        # Classification token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # GFNet blocks (replaces Transformer blocks)
        self.blocks = nn.ModuleList([GFNetBlock(embed_dim, mlp_dim) for _ in range(depth)])

        # Final normalization and classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding (same as ViT)
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        # Append the CLS token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through GFNet blocks
        for block in self.blocks:
            x = block(x)

        # Final layer normalization
        x = self.norm(x)

        # Use the CLS token output for classification
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)


model = GFNet(img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, mlp_dim=3072)

dummy_input = torch.randn(8, 3, 224, 224)
output = model(dummy_input)
print(f"{output.shape == (8, 1000)}")
