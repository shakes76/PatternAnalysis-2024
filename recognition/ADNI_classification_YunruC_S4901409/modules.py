'''
The idea of the code is from the paper: 
Implementing Vision transformer from Scratch 
https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
'''

import torch
import torch.nn as nn
import torch.fft
from timm.models.layers import to_2tuple

class PatchEmbedding(nn.Module):
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

    def forward(self, x):
        B, C, H, W= x.shape
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm='ortho')
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
        self.act = nn.GELU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    


