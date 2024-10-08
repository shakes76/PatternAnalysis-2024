## 1. “modules.py" containing the source code of the components of your model. Each component must be
# implemented as a class or a function
import torch
import torch.nn as nn
from torch.fft import rfft2, irfft2
from timm.models.layers import to_2tuple

class GFNet(nn.Module):
    def __init__(self, dim, W, H, channels, num_classes=2) -> None:
        super().__init__()
        self.dim = dim
        self.width = W
        self.height = H
        self.num_classes = num_classes
        self.channels = channels
        ## Patch embeddings 

        self.embedded_patches = PatchEmbed(img_size=:)

    def forward(self, x):
        pass

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) # type: ignore
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) # type: ignore

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# NOTE: Maybe reshape the input before and after? - got it
# CONSIDER ADDING DROPOUT
class FeedFowardNetwork(nn.Module):
    def __init__(self, dim, W, H, act_layer=nn.LeakyReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.width = W
        self.height = H
        self.norm_layer = norm_layer
        self.act_layer = act_layer()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.norm_layer(x) # WARNING: This might be wrong - don't really know what norm_layer takes as input
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.norm_layer(x)
        x = self.fc2(x)
        x = self.act_layer(x)
        return x

class GlobalFilterLayer(nn.Module):
    def __init__(self, dim, h, w, norm_layer=nn.LayerNorm):
        self.dim = dim
        self.norm_layer = norm_layer
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        super().__init__()

    def forward(self, x):
        B, H, W, C = x.shape

        # x = x.to(torch.float32)

        x = rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
        return x

