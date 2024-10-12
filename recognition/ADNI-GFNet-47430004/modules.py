import torch
import torch.nn as nn

# Hyperparameters

class mlp(nn.Module):
    pass

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        pass

    def forward(self, x, spatial_size=None):
        pass

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        pass

    def forward(self, x):
        pass

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        pass

    def forward(self, x):
        pass

class GFNet(nn.Module):
    # Patch embed
    pass

def main():
    print("Main")