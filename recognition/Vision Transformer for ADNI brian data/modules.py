import math
#import numpy.lib.arraypad as pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from collections import OrderedDict



## Using RaoyonGming GFNet github repo as starting point for model architecture code
## https://github.com/raoyongming/GFNet/blob/master/gfnet.py

class GFNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                embed_dim=783, depth=12, ratio=4., uniform_drop=None, representation_size=None, dropout=0.3, dropout_path=0.1):
        super().__init__()
        num_classes=1
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            dpr = [dropout_path for _ in range(depth)]  
        else:
            dpr = [x.item() for x in torch.linspace(0, dropout_path, depth)] 

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=ratio,
                drop=dropout, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        x = x.squeeze()
        return x

### Feed forward section of GF network
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, dropout=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fnc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fnc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fnc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fnc2(x)
        x = self.drop(x)
        return x

# Global Filter for transformer
class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N)) 
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, dropout=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

""" 
    Image to Patch Embedding
"""
class PatchEmbed(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H == self.img_size[0] and W == self.img_size[1]), \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# class BlockLayerScale(nn.Module):
#         def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
#                     norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
#             super().__init__()
#             self.norm1 = norm_layer(dim)
#             self.filter = GlobalFilter(dim, h=h, w=w)
#             self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#             self.norm2 = norm_layer(dim)
#             mlp_hidden_dim = int(dim * mlp_ratio)
#             self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#             self.gamma = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#         def forward(self, x):
#             x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
#             return x

# class DownLayer(nn.Module):
#     """ Image to Patch Embedding
#     """
#     def __init__(self, img_size=56, dim_in=64, dim_out=128):
#         super().__init__()
#         self.img_size = img_size
#         self.dim_in = dim_in
#         self.dim_out = dim_out
#         self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
#         self.num_patches = img_size * img_size // 4
#     def forward(self, x):
#         B, N, C = x.size()
#         x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
#         x = self.proj(x).permute(0, 2, 3, 1)
#         x = x.reshape(B, -1, self.dim_out)
#         return x