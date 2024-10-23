import torch
import torch.nn as nn
import math
from torch.fft import rfft2, irfft2
from collections import OrderedDict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class GFNet(nn.Module):
    """
    the GFNet module use Global Filters and Transformer Blocks.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2, embed_dim=768, depth=12,
                ff_ratio=4., norm_layer=None, dropout=0.3, drop_path_rate=0.3):
        super().__init__()