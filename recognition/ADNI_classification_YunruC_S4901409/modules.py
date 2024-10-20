import torch
import torch.nn as nn
import torch.fft
from timm.models.layers import to_2tuple, trunc_normal_
import math

'''
The code is from the paper: 
Implementing Vision transformer from Scratch 
https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
and 
the paper:
Global Filter Networks for Image Classification
https://github.com/raoyongming/GFNet/blob/master/gfnet.py
'''

class PatchEmbed(nn.Module):
    """
    Patch the input image and flatten into a 1D sequence.
    embed_dim = embedding dimensionality, it means that each patch is represented 
    by a vectoe of 768 values.
    """

    def __init__(self, img_size, patch_size, input_chans=3, embed_dim=768):
        super().__init__()
        # using the to_2tuple to get the image and patch size in the form of (s1 x s2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1]//patch_size[1])*(img_size[0]//patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(input_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1,2)
        return x

"""
The GFNet does not require explicit position embedding like traditional ViT as
it uses the Global Filters to process the entire input image in the frequency 
domain.

The Multi-Attention in ViT is replaced by Global Filters.
"""

class GFNetFilter(nn.Module):
    """
    The Global Filter layer is copied from the https://gfnet.ivg-research.xyz/
    with some modifications
    """
    def __init__(self, dim , h= 14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size =None):
        B, N, C= x.shape
        if spatial_size is None:
            a = b= int(math.sqrt(N))
        else:
            a = b= spatial_size
        
        x = x.view(B,a,b,C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1,2), norm='ortho')

        x = x.reshape(B, N, C)
        return x
    
class MLP(nn.Module):
    """
    Multilayer perceptron
    """

    def __init__(self, in_features, hidden_features = None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1= nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    """
    A single GFNet block
    dim = number of channels
    mlp_ratio controls the expansion size of the hiden layer
    """

    def __init__ (self, dim, mlp_ratio=4, drop = 0., h =14, w = 8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.filter = GFNetFilter(dim, h=h, w=w)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class GFNet(nn.Module):
    """
    GFNet model
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, input_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, h=h, w=w)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)
        
        #classifier head
        self.head = nn.Linear(self.num_features, num_classes)

        #initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        m is a layer of the neural network.
        For layer type of nn.Linear, the weights are initialized using the trunc_normal_
        function.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


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
        x = self.head(x)
        return x





