"""
This script defines the core modules and components for the GFNet model architecture, including custom layers
and building blocks for deep learning models, specifically for binary classification (AD vs NC).
It includes modules for global filtering, multi-layer perceptrons (MLPs), patch embedding, and transformer blocks,
as well as the overall GFNet model.

The script also provides utility functions for initialising model weights and managing model configurations.

@brief: Core modules and architecture definition for the GFNet model.
@date: 16 Oct 2024
@author: Sean Bourchier
"""

import math
import torch
import torch.fft
import torch.nn as nn
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from dataset import ADNI_DEFAULT_MEAN_TEST, ADNI_DEFAULT_STD_TEST

def _cfg(url='', **kwargs):
    """
    Helper function to create model configuration.
    
    Args:
        url (str): URL for pretrained model weights.
        **kwargs: Additional configuration parameters.
        
    Returns:
        dict: Configuration dictionary for model initialization.
    """
    return {
        'url': url,
        'num_classes': 2, 
        'input_size': (1, 224, 224), 
        'pool_size': None,
        'crop_pct': 0.9, 
        'interpolation': 'bicubic',
        'mean': ADNI_DEFAULT_MEAN_TEST, 
        'std': ADNI_DEFAULT_STD_TEST,
        'first_conv': 'patch_embed.proj', 
        'classifier': 'head',
        **kwargs
    }

class Mlp(nn.Module):
    """
    Multi-layer perceptron (MLP) block.
    Applies a linear transformation followed by a GELU activation and dropout.
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            act_layer (nn.Module): Activation function (default: GELU).
            drop (float): Dropout rate (default: 0.0).
        """
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after linear, activation, and dropout layers.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    """
    A global filtering layer using 2D FFT and learnable complex weights.
    """

    def __init__(self, dim, h=14, w=8):
        """
        Args:
            dim (int): Dimension of the input features.
            h (int): Height of the FFT filter.
            w (int): Width of the FFT filter.
        """
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        """
        Forward pass through the global filter.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            spatial_size (tuple, optional): Spatial size of the input (height, width).

        Returns:
            torch.Tensor: Output tensor after global filtering.
        """
        B, N, C = x.shape
        a, b = spatial_size if spatial_size else (int(math.sqrt(N)), int(math.sqrt(N)))

        x = x.view(B, a, b, C).to(torch.float32)

        # Apply 2D FFT, filtering with complex weights, and inverse FFT
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        return x.reshape(B, N, C)

class Block(nn.Module):
    """
    Transformer block with normalization, global filtering, and MLP layers.
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        """
        Args:
            dim (int): Dimension of the input features.
            mlp_ratio (float): Ratio of MLP hidden dimension to input dimension.
            drop (float): Dropout rate.
            drop_path (float): Drop path rate.
            act_layer (nn.Module): Activation function.
            norm_layer (nn.Module): Normalization layer.
            h (int): Height for the global filter.
            w (int): Width for the global filter.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Forward pass through the block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after the block.
        """
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class BlockLayerScale(Block):
    """
    Transformer block with learnable scaling parameter for the MLP output.
    Inherits from the Block class.
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5):
        """
        Args:
            dim (int): Dimension of input features.
            init_values (float): Initial values for the learnable scaling parameter.
            Other parameters as in the Block class.
        """
        super().__init__(dim, mlp_ratio, drop, drop_path, act_layer, norm_layer, h, w)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """
        Forward pass with layer scaling.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying the block with scaling.
        """
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x

class PatchEmbed(nn.Module):
    """
    Converts an image into patch embeddings.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Size of each patch.
            in_chans (int): Number of input channels.
            embed_dim (int): Dimension of the output embedding.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass for patch embedding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Patch embeddings of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DownLayer(nn.Module):
    """
    Reduces the spatial dimensions of an image using a convolutional layer.
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        """
        Args:
            img_size (int): Input image size.
            dim_in (int): Input channel dimension.
            dim_out (int): Output channel dimension.
        """
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        """
        Forward pass for downsampling.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C).
            
        Returns:
            torch.Tensor: Downsampled tensor.
        """
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)
        return x

class GFNet(nn.Module):
    """
    GFNet model with patch embedding, positional embeddings, multiple transformer blocks, 
    and a classification head.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2, embed_dim=768, depth=12,
                 mlp_ratio=4., uniform_drop=False, drop_rate=0., drop_path_rate=0., 
                 norm_layer=None, dropcls=0):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input channels.
            num_classes (int): Number of classes for classification.
            embed_dim (int): Embedding dimension.
            depth (int): Number of transformer blocks.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            uniform_drop (bool): Whether to use uniform drop path.
            drop_rate (float): Dropout rate.
            drop_path_rate (float): Drop path rate for stochastic depth.
            norm_layer (nn.Module): Normalization layer.
            dropcls (float): Dropout rate before classifier.
        """
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        dpr = [drop_path_rate] * depth if uniform_drop else [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], 
                  norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)

        # Representation layer for pre-logits
        self.pre_logits = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(embed_dim, embed_dim)),
            ('act', nn.Tanh())
        ])) if embed_dim else nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.final_dropout = nn.Dropout(p=dropcls) if dropcls > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize weights of the model.
        
        Args:
            m (nn.Module): The module whose weights are to be initialized.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """
        Extract features from the input image.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Feature tensor.
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
        """
        Forward pass through the entire model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output logits.
        """
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x
