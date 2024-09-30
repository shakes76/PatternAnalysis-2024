import torch
import torch.nn as nn
from GFNet.gfnet import GFNet, GFNetPyramid
from functools import partial

class GFNetBinaryClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(GFNetBinaryClassifier, self).__init__()
        self.model = GFNetPyramid(
            img_size=224, num_classes=num_classes,
            patch_size=4, embed_dim=[96, 192, 384, 768], depth=[3, 3, 27, 3],
            mlp_ratio=[4, 4, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.4, init_values=1e-6
        )

    def forward(self, x):
        return self.model(x)