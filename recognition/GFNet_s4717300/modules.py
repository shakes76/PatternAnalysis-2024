'''
@file   modules.py
@brief  Contains GFNet Architecture
@author  Benjamin Jorgensen - s4717300
@date   18/10/2024
'''
import torch
import torch.nn as nn
import math
from torch.fft import rfft2, irfft2
from collections import OrderedDict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class GFNet(nn.Module):
    """
    Main GFNet module, initialise with all hyperparameters.

    @param img_size: size in pixels of the training data image (must be square)
    @param patch_size: value for the width and hight for each patch in pixels
    @param in_chans: Colour channels of each image, usually greyscale for ANDI
    @param num_classes: Number of classes for the classifier to classify
    @param embed_dim: number of dimensions to convert a patch to
    @param depth: number of global filter blocks in the network
    @param ff_ratio: The ratio of input to hidden layers in the feed forward network
    @param norm_layer: Normalisation function used between layers of the global filter
    @param dropout: Probability of neuron dropout in the network for a iteration, revering the neuron inactive.
    @param drop_path: Probability of neural pathway deactivating for a given iteration.
    """
    def __init__(self, img_size, patch_size, in_chans=3, num_classes=10, embed_dim=768, depth=12,
                 ff_ratio=4., norm_layer=None, dropout=0.3, drop_path_rate=0.3):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        h = img_size // patch_size
        w = h // 2 + 1

        # Creating drop path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Defining the blocks in each layer
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, ff_ratio=ff_ratio,
                norm_layer=norm_layer, h=h, w=w, drop=dropout, drop_path=dpr[i]) #type: ignore
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Create initial weights for neurons
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Create initial weights for each neuron
        @params m: neural layer
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore #type: ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        Feed forward some inputs to the next layer
        @param x: input
        """
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
        # x = self.final_dropout(x)
        x = self.head(x)
        return x

class Block(nn.Module):
    """
    A layer in the GFNet including the global filter.
    """
    def __init__(self, dim, ff_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ff_ratio)
        self.feedForward = FeedFowardNetwork(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.feedForward(self.norm2(self.filter(self.norm1(x)))))
        return x

class PatchEmbed(nn.Module):
    """ 
    Image to Patch Embedding. Splits the image based on patch size and turns
    that into an embedding, (vector representation). Uses a 2D convolution
    layer with step size equal to patch size equal to kernel size.
    """
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


class FeedFowardNetwork(nn.Module):
    """
    Classification module of the neural network after features have been
    learned from the global filters.

    @param in_features: number of inputs
    @param hidden_features: number of neurons in the hidden layer
    @param out_features: number of neurons in the final hidden layer
    @param act_layer: Activation function of the feed forward neural network
    @param drop: probability of deactivating a neuron for the given iteration
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act_layer = act_layer()
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    """
    The Global filter is responsible for learning the features of the image. It
    uses attention mechanism like a vision transformer but it uses the Fourier
    space for the features, reducing complexity with desirable performance/
    complexity trade-offs.

    @params dim: dimensions of the input data
    @params w: width of the input data in pixels
    @params h: height of the input data in pixels
    """
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

        x = rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        return x