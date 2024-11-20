"""
modules.py

Source code of the components of the vision transformer.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
import torch.nn as nn
import math

from constants import DROPOUT_RATE
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class GlobalFilterLayer(nn.Module):
    """
    Layer that applies a learnable global filter to the input using Fourier Transform.
    The layer applies a filter in the frequency domain (Fourier space) to the input tensor by 
    transforming it to the frequency domain, applying a complex weight, and then transforming 
    it back to the spatial domain.
    """
    def __init__(self, dim, h=14, w=8):
        """
        Initialise the layer with learnable complex weights.
        
        Args:
            dim (int): The input feature dimension.
            h (int): Height of the input tensor for reshaping.
            w (int): Width of the input tensor for reshaping.
        """
        super().__init__()

        # Initialize a complex weight parameter (real and imaginary parts)
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        """
        Forward pass that applies the global filter on the input using Fourier Transforms.
        
        Args:
            x (Tensor): Input tensor of shape (B, N, C) where B is the batch size, 
                        N is the number of patches, and C is the feature dimension.
            spatial_size (tuple, optional): The spatial size of the input if provided, 
                                             otherwise it is inferred from the input shape.
        
        Returns:
            Tensor: Output tensor after applying the global filter.
        """
        B, N, C = x.shape

        # If spatial size is not provided, infer it from N (assuming square spatial dimensions)
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        # Apply 2D FFT to transform the input to frequency space
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)

        # Apply the learned complex weight in frequency space
        x = x * weight

        # Inverse FFT to get back to spatial space
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) that consists of linear layers, activation functions, 
    and dropout for regularization.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=DROPOUT_RATE):
        """
        Initialise the MLP with specified dimensions and regularization.
        
        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to `in_features`.
            out_features (int, optional): Number of output features. Defaults to `in_features`.
            act_layer (nn.Module, optional): Activation function layer (default is GELU).
            drop (float, optional): Dropout rate (default is defined DROPOUT_RATE constant).
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass of the MLP block.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying the MLP block.
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    """
    Transformer-style block that consists of a GlobalFilterLayer followed by an MLP block.
    The block uses normalization, residual connections, and optional dropout and path drop.
    """
    def __init__(self, dim, mlp_ratio=4., drop=DROPOUT_RATE, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, h=14, w=8):
        """
        Initialise the transformer block with a global filter layer, MLP, and normalization.
        
        Args:
            dim (int): The input feature dimension.
            mlp_ratio (float): Multiplier for the size of the hidden layer in the MLP.
            drop (float): Dropout rate for the MLP.
            drop_path (float): DropPath rate for stochastic depth.
            act_layer (nn.Module): The activation function for MLP.
            norm_layer (nn.Module): The normalization layer.
            h (int): Height of the input tensor for reshaping.
            w (int): Width of the input tensor for reshaping.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilterLayer(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after processing through the block.
        """
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x
    
class PatchEmbed(nn.Module):
    """
    Patch embedding layer that divides an image into patches and projects them to a higher dimensional space.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, embed_dim=768):
        """
        Initialise the patch embedding layer.
        
        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of the patch.
            in_chans (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB).
            embed_dim (int): The dimension of the embedding space.
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Convolution to project the image patches to the embedding dimension
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass that performs the patch embedding.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W), where B is batch size,
                        C is number of channels, H is height, and W is width.
        
        Returns:
            Tensor: Flattened patches of shape (B, N, C) where N is the number of patches.
        """
        B, C, H, W = x.shape

        # Check if input image size matches the expected size
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class GFNet(nn.Module):
    """
    Vision transformer-style model GFNet that uses global filter layers combined with MLP blocks 
    to process input images and classify them.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1, num_classes=2, embed_dim=384, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=DROPOUT_RATE, drop_path_rate=0., norm_layer=None, 
                 dropcls=0):
        """
        Initialise the GFNet model with the specified parameters.
        
        Args:
            img_size (int): The size of the input image.
            patch_size (int): The size of each patch.
            in_chans (int): The number of input channels.
            num_classes (int): The number of output classes for classification.
            embed_dim (int): The dimensionality of the embeddings.
            depth (int): The number of transformer blocks.
            mlp_ratio (float): The ratio of hidden dimension to input dimension in the MLP.
            representation_size (int, optional): The size of the representation layer for bottleneck.
            uniform_drop (bool): Whether to apply uniform dropout across layers.
            drop_rate (float): Dropout rate.
            drop_path_rate (float): Drop path rate for stochastic depth.
            norm_layer (nn.Module): The normalization layer.
            dropcls (float): Dropout rate for the classification layer.
        """
        super().__init__()

        # Set number of output classes and embedding dimension
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # For consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) # Use LayerNorm by default

        # Initialise the patch embedding layer
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # Number of patches from input image

        # Positional embedding parameters
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # Learnable positional encoding
        self.pos_drop = nn.Dropout(p=drop_rate) # Dropout applied to positional embeddings

        # Determine image height and width after patch splitting
        h = img_size // patch_size

        # Adjusted width after patch splitting
        w = h // 2 + 1

        # Configure stochastic depth (drop path) rates based on uniform or linear decay
        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # Uniform drop path rate for all layers
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Linear decay of drop path rate
        
        # Initialise transformer blocks (GlobalFilterLayer + MLP)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, mlp_ratio=mlp_ratio,
                drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])
        
        # Final normalisation layer
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head (linear layer for classification)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Dropout before the classifier for regularisation
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        # Initialise weights
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Weight initialisation function for various layers.
        
        Args:
            m (nn.Module): The layer module (either Linear or LayerNorm).
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02) # Truncated normal initialisation for linear layer weights
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0) # Initialise bias to zero
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0) # Initialise LayerNorm bias to zero
            nn.init.constant_(m.weight, 1.0) # Initialise LayerNorm weight to 1

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Returns the list of parameters that should not have weight decay applied (e.g., pos_embed, cls_token).
        
        Returns:
            set: Set of parameters to exclude from weight decay.
        """
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        """
        Get the classifier head (final linear layer).
        
        Returns:
            nn.Module: The classifier head module.
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        """
        Reset the classifier head with a new number of output classes.
        
        Args:
            num_classes (int): The number of output classes for classification.
            global_pool (str): Optional argument for global pooling (unused).
        """
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        Forward pass through the feature extraction layers (patch embedding + transformer blocks).
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels,
                        H is the height, and W is the width.
        
        Returns:
            Tensor: The output tensor after passing through the feature extraction layers.
        """
        B = x.shape[0]
        x = self.patch_embed(x) # Convert image to patches and embed
        x = x + self.pos_embed # Add positional encoding to embedded patches
        x = self.pos_drop(x) # Apply dropout to positional encoding

        # Pass through each transformer block
        for blk in self.blocks:
            x = blk(x)

        # Apply normalisation and pooling (mean across patches)
        x = self.norm(x).mean(1)
        return x

    def forward(self, x):
        """
        Forward pass through the entire GFNet model.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels,
                        H is the height, and W is the width.
        
        Returns:
            Tensor: The output tensor with predictions from the classifier head.
        """
        x = self.forward_features(x) # Extract features using forward_features
        x = self.final_dropout(x) # Apply dropout before the classifier
        x = self.head(x) # Classify using the final head
        return x