""" This file contains the source code of the components of this model.
Each component is implemented as a class or in a function.

The general structure of this vision Transformer was made in assistance by following 
these sources:
-   Shengjie, Z., Xiang, C., Bohan, R., & Haibo, Y. (2022, August 29). 
    3D Global Fourier Network for Alzheimer’s Disease Diagnosis using Structural MRI. MICCAI 2022
    Accepted Papers and Reviews. https://conferences.miccai.org/2022/papers/002-Paper1233.html

‌
"""
import torch
import torch.nn as nn
import torch.fft
from functools import partial
from collections import OrderedDict
import math
from timm.models.layers import DropPath, trunc_normal_ # extra library for improving training

# MLP Block similar to the example
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # Apply 2 linear units with drop out
        self.fc1 = nn.Linear(in_features, hidden_features)
        # set activation to GELU (used in ViT)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
# Global Filter following the FFT-based logic
class Global_Filter(nn.Module):
    def __init__(self, h=14, w=8, dim=1000):
        super().__init__()
        # Learnable complex weight parameter for FFT
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim,2 , dtype=torch.float32) * 0.02)
        self.dim = dim
        self.h = h
        self.w = w

    def forward(self, x, spatial_size = None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        # Forward FFT
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        # Inverse FFT
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)
        return x
    

# Block with Global Filter and MLP
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.5, drop_path=0.6, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = Global_Filter(dim=dim, h=h, w=w)
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


# Patch embedding from image to flattened patches
class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim,kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# GFNet main, follows block stacking with dropout
class GFNet(nn.Module):
    #images are set 256 x 256 with RGB # change depth ect
    def __init__(self, img_size=256, patch_size= 16, embed_dim=768, num_classes=2, in_channels=3, drop_rate=0.5, depth=8, mlp_ratio=4., drop_path_rate=0.6, norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # call the patchembed to flatten patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        h = img_size // patch_size
        w =  h // 2 + 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        # Classification head for Alzheimer's (AD vs NC)
        self.head = nn.Linear(self.num_features, num_classes)
        self.apply(self._init_weights)

    # Now check where it belongs and apply the constants
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Clip values outside 2 stds from the mean (of normal distribution)
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


