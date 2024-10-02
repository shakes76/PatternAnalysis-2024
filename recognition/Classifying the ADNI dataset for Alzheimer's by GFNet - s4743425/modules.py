""" This file contains the source code of the components of this model.
Each component is implemented as a class or in a function.

The general structure of this vision Transformer was made in assistance by following 
these sources:

Shengjie, Z., Xiang, C., Bohan, R., & Haibo, Y. (2022, August 29). 
3D Global Fourier Network for Alzheimer’s Disease Diagnosis using Structural MRI. MICCAI 2022
 - Accepted Papers and Reviews. https://conferences.miccai.org/2022/papers/002-Paper1233.html

‌
"""
import torch
import torch.nn as nn
import torch.fft
from functools import partial
from collections import OrderedDict

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
        x = self.act(x)  
        x = self.drop(x)
        return x
    
# Global Filter following the FFT-based logic
class Global_Filter(nn.Module):
    def __init__(self, h=9, w=10, d=5, dim=1000):
        super().__init__()
        # Learnable complex weight parameter for FFT
        self.complex_weight = nn.Parameter(torch.randn(
            h, w, d//2+1, dim, 2, dtype=torch.float32) * 0.02)
        self.dim = dim
        self.h = h
        self.w = w
        self.d = d

    def forward(self, x):
        B, N, C = x.shape
        x = x.to(torch.float32)
        x = x.view(B, self.h, self.w, self.d, self.dim)
        # Forward FFT
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        # Inverse FFT
        x = torch.fft.irfftn(x, s=(self.h, self.w, self.d), dim=(1, 2, 3), norm='ortho')
        x = x.reshape(B, N, C)
        return x
    

# Block with Global Filter and MLP
class Block(nn.Module):
    def __init__(self, dim=1000, mlp_ratio=2., drop=0.5, drop_path=0.6, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=18, w=21, d=10):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = Global_Filter(dim=dim, h=h, w=w, d=d)
        # setting to Identity regardless, can include a drop path later
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
    def __init__(self, img_size=(181, 217, 181), patch_size=(10, 10, 10), num_classes=2, in_channels=1):
        super().__init__()
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] //
                                                        patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_dim = in_channels * \
            patch_size[0]*patch_size[1]*patch_size[2]
        self.proj = nn.Conv3d(in_channels, self.patch_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        N, C, H, W, D = x.size()
        assert H == self.img_size[0] and W == self.img_size[1] and D == self.img_size[2],\
            f"Input image size ({H}*{W}*{D}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# GFNet main, follows block stacking with dropout
class GFNet(nn.Module):
    def __init__(self, img_size=(181, 217, 181), patch_size=(10, 10, 10), embed_dim=1000, num_classes=2, in_channels=1, drop_rate=0.5, depth=1, mlp_ratio=2., drop_path_rate=0.6, norm_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # call the patchembed to flatten patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        h, w, d = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w, d=d)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )
        self.apply(self._init_weights)

    # Now check where it belongs and apply the constants
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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

#### Include some form of truncated normal distribution to tensors

if __name__ == '__main__':
    # test to see if it works
    x = torch.randn(2, 1, 256, 256, 256)
    net = GFNet(img_size=[256, 256, 256], patch_size=[16, 16, 16], embed_dim=4096)
    output = net(x)
    print(output.shape)