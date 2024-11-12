import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
from functools import partial
from collections import OrderedDict

class Mlp(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) module.

    Args:
        in_features (int): Number of input features.
        hidden_features (int, optional): Number of hidden features. Defaults to in_features.
        out_features (int, optional): Number of output features. Defaults to in_features.
        act_layer (nn.Module, optional): Activation layer to use. Defaults to nn.GELU.
        drop (float, optional): Dropout rate. Defaults to 0.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        act (nn.Module): Activation layer.
        fc2 (nn.Linear): Second fully connected layer.
        drop (nn.Dropout): Dropout layer.

    Methods:
        forward(x):
            Forward pass of the MLP.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after applying the MLP.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the neural network module.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GlobalFilter(nn.Module):
    """
    A PyTorch module that applies a global filter to the input tensor using FFT.

    Args:
        dim (int): The dimension of the input tensor.
        h (int, optional): The height of the complex weight tensor. Default is 14.
        w (int, optional): The width of the complex weight tensor. Default is 8.

    Attributes:
        complex_weight (torch.nn.Parameter): A learnable parameter representing the complex weights.
        h (int): The height of the complex weight tensor.
        w (int): The width of the complex weight tensor.

    Methods:
        forward(x, spatial_size=None):
            Applies the global filter to the input tensor.
            Args:
                x (torch.Tensor): The input tensor of shape (B, N, C).
                spatial_size (tuple, optional): The spatial size (height, width) of the input tensor. If None, it is inferred from the input tensor.
            Returns:
                torch.Tensor: The filtered tensor of shape (B, N, C).
    """
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2) * 0.02)
        self.h = h
        self.w = w

    def forward(self, x, spatial_size=None):
        """
        Perform the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, N, C), where B is the batch size,
                              N is the number of spatial locations, and C is the number of channels.
            spatial_size (tuple, optional): Tuple (h, w) representing the height and width of the spatial dimensions.
                                            If None, the spatial dimensions are assumed to be square with size sqrt(N).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input (B, N, C).
        """
        B, N, C = x.shape
        if spatial_size is None:
            h = w = int(math.sqrt(N))
        else:
            h, w = spatial_size

        x = x.view(B, h, w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(h, w), dim=(1, 2), norm='ortho')
        x = x.view(B, N, C)
        return x

class Block(nn.Module):
    """
    A neural network block that applies a normalisation layer, a global filter, 
    and a multi-layer perceptron (MLP) with optional dropout and drop path.

    Args:
        dim (int): The dimension of the input tensor.
        mlp_ratio (float, optional): The ratio of the hidden layer size to the input size in the MLP. Default is 4.0.
        drop (float, optional): Dropout rate. Default is 0.0.
        drop_path (float, optional): Drop path rate. Default is 0.0.
        act_layer (nn.Module, optional): Activation layer to use in the MLP. Default is nn.GELU.
        norm_layer (nn.Module, optional): Normalisation layer to use. Default is nn.LayerNorm.
        h (int, optional): Height parameter for the GlobalFilter. Default is 14.
        w (int, optional): Width parameter for the GlobalFilter. Default is 8.

    Attributes:
        norm1 (nn.Module): The normalisation layer.
        filter (GlobalFilter): The global filter layer.
        drop_path (nn.Module): The drop path layer.
        mlp (nn.Sequential): The multi-layer perceptron (MLP) with normalisation and activation.

    Methods:
        forward(x):
            Forward pass of the block. Applies normalisation, global filter, and MLP with optional dropout and drop path.
            Args:
                x (torch.Tensor): Input tensor.
            Returns:
                torch.Tensor: Output tensor after applying the block operations.
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = nn.Sequential(
            norm_layer(dim),
            Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        )

    def forward(self, x):
        """
        Perform a forward pass through the network module.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after applying the network transformations.
        """
        x = x + self.drop_path(self.filter(self.norm1(x)))
        x = x + self.drop_path(self.mlp(x))
        return x

class PatchEmbed(nn.Module):
    """
    Patch Embedding Layer for Vision Transformers.

    Args:
        img_size (int or tuple): Size of the input image. Default is 224.
        patch_size (int or tuple): Size of the patches to be extracted from the input image. Default is 16.
        in_chans (int): Number of input channels. Default is 3.
        embed_dim (int): Dimension of the embedding. Default is 768.

    Attributes:
        num_patches (int): Number of patches extracted from the input image.
        proj (nn.Conv2d): Convolutional layer to project the input image into patch embeddings.

    Methods:
        forward(x):
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_chans, height, width).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, num_patches, embed_dim).
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass for the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after projection, flattening, and transposition.
        """
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GFNet(nn.Module):
    """
    GFNet is a neural network model for image classification tasks.

    Args:
        img_size (int): Size of the input image. Default is 224.
        patch_size (int): Size of the patches to split the image into. Default is 16.
        in_chans (int): Number of input channels. Default is 3.
        num_classes (int): Number of output classes. Default is 1000.
        embed_dim (int): Dimension of the embedding. Default is 1024.
        depth (int): Number of blocks in the network. Default is 16.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default is 4.0.
        drop_rate (float): Dropout rate. Default is 0.1.
        drop_path_rate (float): Drop path rate. Default is 0.2.
        norm_layer (nn.Module): Normalisation layer. Default is nn.LayerNorm.

    Attributes:
        num_classes (int): Number of output classes.
        patch_embed (PatchEmbed): Patch embedding layer.
        pos_embed (nn.Parameter): Positional embedding.
        pos_drop (nn.Dropout): Dropout layer for positional embedding.
        blocks (nn.Sequential): Sequence of transformer blocks.
        norm (nn.Module): Normalisation layer.
        head (nn.Linear or nn.Identity): Classification head.

    Methods:
        _init_weights(m): Initializes the weights of the model.
        forward(x): Forward pass of the model.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=1024, depth=16,
                 mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=img_size // patch_size, w=img_size // (2 * patch_size))
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialise the weights of the given module.

        Parameters:
        m (nn.Module): The module to initialise. This can be an instance of nn.Linear or nn.LayerNorm.

        For nn.Linear:
            - The weights are initialised using a truncated normal distribution with a standard deviation of 0.02.
            - The biases are initialised to 0 if they are not None.

        For nn.LayerNorm:
            - The biases are initialised to 0.
            - The weights are initialised to 1.0.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x).mean(1)
        return self.head(x)