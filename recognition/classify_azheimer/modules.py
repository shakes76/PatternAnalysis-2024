import math
from functools import partial
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
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


class GlobalFilter(nn.Module):
    def __init__(self, dim, h, w):
        super().__init__()
        
        self.w = w
        self.h = h
        self.complex_weight = nn.Parameter(
            torch.randn(self.h, w // 2 + 1, dim, 2, dtype=torch.float32) * 0.02
        )
        
    def forward(self, x):
        B, N, C = x.shape
        a, b = self.h, self.w

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        x = x.reshape(B, N, C)

        return x


class BlockLayerScale(nn.Module):

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        h=14,
        w=8,
        init_values=1e-5,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(
            self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x))))
        )
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DownLayer(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size, dim_in, dim_out):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = (img_size[0] // 2) * (img_size[1] // 2)

    def forward(self, x):
        B, N, C = x.size()
        H, W = self.img_size
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x


class GFNetPyramid(nn.Module):

    def __init__(
        self,
        img_size=(256, 240),
        patch_size=4,
        num_classes=2,
        embed_dim=[96, 192, 384, 768],
        depth=[3, 3, 27, 3],
        mlp_ratio=[4, 4, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.4,
        init_values=1e-6,
        dropcls=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim[
            -1
        ]  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = nn.ModuleList()

        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim[0]
        )
        num_patches = patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))

        self.patch_embed.append(patch_embed)

        sizes = [(256 // (2**i), 240 // (2**i)) for i in range(2, 6)]
        for i in range(3):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i + 1])
            self.patch_embed.append(patch_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))
        ]  # stochastic depth decay rule
        cur = 0
        for i in range(4):
            w,h = sizes[i]
            blk = nn.Sequential(
                *[
                    BlockLayerScale(
                        dim=embed_dim[i],
                        mlp_ratio=mlp_ratio[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer,
                        h=h,
                        w=w,
                        init_values=init_values,
                    )
                    for j in range(depth[i])
                ]
            )
            self.blocks.append(blk)
            cur += depth[i]

        # Classifier head
        self.norm = norm_layer(embed_dim[-1])

        self.head = nn.Linear(self.num_features, num_classes)

        if dropcls > 0:
            print("dropout %.2f before classifier" % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(4):
            x = self.patch_embed[i](x)
            if i == 0:
                x = x + self.pos_embed
            x = self.blocks[i](x)

        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x
