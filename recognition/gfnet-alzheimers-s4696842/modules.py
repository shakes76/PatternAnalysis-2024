import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """
    A simple Multilayer Perceptron (MLP) block consisting of two linear layers
    with an activation function and dropout.
    """

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
    """
    A global filter layer that performs filtering in the Fourier domain.

    This layer transforms the input tensor into the frequency domain using a 2D Fourier transform,
    applies a learnable complex filter, and then transforms it back to the spatial domain.

    Args:
        dim (int): Embedding dimension.
        h (int): Height dimension after patch embedding.
        w (int): Width dimension in the frequency domain after patch embedding.
    """

    def __init__(self, dim, h, w):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.h = h

    def forward(self, x):
        B, N, C = x.shape
        a = self.h
        b = N // a
        x = x.view(B, a, b, C).to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")
        x = x.reshape(B, N, C)
        return x


class Block(nn.Module):
    """
    A transformer block that includes a GlobalFilter layer, normalization layers, and an MLP.

    Args:
        dim (int): Embedding dimension.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.0.
        drop (float, optional): Dropout rate after MLP layers. Defaults to 0.0.
        drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
        act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        h (int, optional): Height dimension for the GlobalFilter. Defaults to 14.
        w (int, optional): Width dimension for the GlobalFilter. Defaults to 8.
    """

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

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding

    Splits the input image into non-overlapping patches and embeds them into a sequence of tokens.

    Args:
        img_size (tuple of int, optional): Input image size. Defaults to (224, 224).
        patch_size (tuple of int, optional): Size of each patch. Defaults to (16, 16).
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        embed_dim (int, optional): Embedding dimension for each patch. Defaults to 768.
    """

    def __init__(
        self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class GFNet(nn.Module):
    """
    Global Filter Network (GFNet).

    GFNet is a neural network architecture that captures both local and global features of an image
    by combining patch embeddings with global filtering in the Fourier domain.

    # Key Components:

    - **Patch Embedding**: Splits the input image into non-overlapping patches and embeds them into a sequence.
    - **Position Embedding**: Adds learnable position embeddings to retain spatial information.
    - **Transformer Blocks**: A series of blocks that include global filtering and MLP layers, with residual connections and normalization.
    - **Classification Head**: A linear layer that maps the extracted features to class logits.

    # Global Filtering:

    The `GlobalFilter` layer performs a 2D Fast Fourier Transform on the input, applies a learnable complex filter,
    and then transforms it back to the spatial domain using an inverse FFT. This allows the model to capture
    global dependencies across the entire image.

    Args:
        img_size (tuple of int, optional): Input image size. Defaults to (256, 240).
        patch_size (tuple of int, optional): Size of each patch. Defaults to (16, 16).
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        num_classes (int, optional): Number of classes for classification. Defaults to 1000.
        embed_dim (int, optional): Embedding dimension. Defaults to 768.
        depth (int, optional): Number of transformer blocks. Defaults to 12.
        mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimension. Defaults to 4.0.
        representation_size (int, optional): Size of the representation layer before the classification head. If None, this layer is not used. Defaults to None.
        uniform_drop (bool, optional): If True, uses a uniform drop path rate across all blocks. If False, uses a linear decay. Defaults to False.
        drop_rate (float, optional): Dropout rate applied after embeddings. Defaults to 0.0.
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        dropcls (float, optional): Dropout rate before the classification head. Defaults to 0.0.
    """

    def __init__(
        self,
        img_size=(256, 240),
        patch_size=(16, 16),
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        representation_size=None,
        uniform_drop=False,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        dropcls=0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h, w_real = self.patch_embed.grid_size
        w_freq = w_real // 2 + 1

        if uniform_drop:
            print("using uniform droppath with expect rate", drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            print("using linear droppath with expect rate", drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    h=h,
                    w=w_freq,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        if dropcls > 0:
            print(f"dropout {dropcls:.2f} before classifier")
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
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
        return x
