"""
A implementation of a Style Generative Adversarial Network 2 (StyleGAN2) designed for 256x240 greyscale images.

REFERENCES:

(1) This code was developed with assistance from the Claude AI assistant,
    created by Anthropic, PBC. Claude provided guidance on implementing
    StyleGAN2 architecture and training procedures.

    Date of assistance: 8/10/2024
    Claude version: Claude-3.5 Sonnet
    For more information about Claude: https://www.anthropic.com

(2) GitHub Repository: stylegan2-ada-pytorch
    URL: https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main
    Accessed on: 29/09/24 - 8/10/24
    
(3) Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). 
    Analyzing and improving the image quality of StyleGAN.
    arXiv. https://arxiv.org/abs/1912.04958

(4) Karras, T., Laine, S., & Aila, T. (2019).
    A Style-Based Generator Architecture for Generative Adversarial Networks.
    arXiv. https://arxiv.org/abs/1812.04948
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils
import numpy as np
import torch.nn.functional as F
import math


class FullyConnectedLayer(nn.Module):
    """
    A flexible fully connected layer with various customisation options.
    Can be used in the mapping network and other parts of StyleGAN2.
    
    Args:
        in_features (int): Num input features.
        out_features (int): Num output features.
        bias (bool, optional): Include bias term? Defaults to True.
        activation (str, optional): Activation function to use. Defaults to 'linear'.
        weight_init (str): Weight initialisation method. Defaults to 'xavier'.
        dropout (float): Dropout rate. Defaults to 0.0.
        batch_norm (bool, optional): Use batch normalisation? Defaults to False.
        layer_norm (bool, optional): Use layer normalisation? Defaults to False.
    """
    def __init__(self, 
                 in_features,
                 out_features,
                 bias=True, 
                 activation='linear', 
                 weight_init='xavier',
                 dropout=0.0, 
                 batch_norm=False,
                 layer_norm=False
                ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        
        # Init weights and biases
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters(weight_init)
        self.act_fn = self.get_activation_fn(activation)

    def reset_parameters(self, weight_init):
        """Initialise the weights using the specified method."""
        if weight_init == 'xavier':
            nn.init.xavier_uniform_(self.weight)
        elif weight_init == 'kaiming':
            nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        elif weight_init == 'orthogonal':
            nn.init.orthogonal_(self.weight)
        else:
            raise ValueError(f"Unsupported weight initialisation: {weight_init}")
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def get_activation_fn(self, activation):
        """Return the specified activation function."""
        if activation == 'relu':
            return F.relu
        elif activation == 'leaky_relu':
            return lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif activation == 'elu':
            return F.elu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'swish':
            return lambda x: x * torch.sigmoid(x)
        elif activation == 'linear':
            return lambda x: x
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def forward(self, x):
        """Forward pass of the fully connected layer."""
        x = F.linear(x, self.weight, self.bias)
        
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.layer_norm:
            x = self.layer_norm(x)
        
        x = self.act_fn(x)
        
        if self.dropout:
            x = self.dropout(x)
        
        return x


class MappingNetwork(nn.Module):
    """
    Conditional Mapping Network for StyleGAN2.
    
    Network to map input latent code z and a label to intermediate latent space w.
    W used to control styles at each layer of synthesis network.

    Args:
        z_dim (int): Dim of input latent code z.
        w_dim (int): Dim of intermediate latent code w.
        num_layers (int): Num layers in mapping network.
        label_dim (int): Dim of label embedding.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    def __init__(self,
                 z_dim, 
                 w_dim,
                 num_layers,
                 label_dim,
                 dropout=0.1
                ):
        super().__init__()
        
        # Label embedding
        self.label_embedding = nn.Embedding(label_dim, z_dim)
        
        layers = []
        for i in range(num_layers):
            layers.append(FullyConnectedLayer(
                z_dim if i == 0 else w_dim,
                w_dim,
                activation='leaky_relu',
                weight_init='kaiming',
                dropout=dropout,
                batch_norm=True
            ))
        self.net = nn.Sequential(*layers)

    def forward(self, z, labels):
        """Transform the input latent code z and labels to the intermediate latent code w."""
        embedded_labels = self.label_embedding(labels)
        z_prime = z + embedded_labels  # Combine latent and label information
        return self.net(z_prime)


class NoiseInjection(nn.Module):
    """
    Noise Injection module for StyleGAN2.
    
    Module adds learnable per-pixel noise to the output of convolutional layers
    in the generator. Helps in generating fine details and stochastic variations
    in the created images.

    Args:
        channels (int): Num input channels.
    """
    def __init__(self, 
                 channels
                ):
        super().__init__()
        # Create a learnable parameter for scaling the noise
        # One scaling factor per channel
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))
    
    def forward(self, x, noise=None):
        """Inject noise in input tensor."""
        if noise is None:
            # Generate random noise if not provided
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device, dtype=x.dtype)
        
        # Scale the noise by the learned weight and add it to the input
        return x + self.weight * noise


class ModulatedConv2d(nn.Module):
    """
    Modulated Convolution layer for StyleGAN2.
    
    Applies style-based modulation to convolution weights.

    Args:
        in_channels (int): Num input channels.
        out_channels (int): Num output channels.
        kernel_size (int): Size of the conv kernel.
        style_dim (int): Dim of style vector.
        demodulate (bool, optional): Use demodulation? Defaults to True.
        up (int, optional): Upsampling factor. Defaults to 1.
        down (int, optional): Downsampling factor. Defaults to 1.
        padding (int, optional): Padding for conv. Defaults to 0.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 style_dim,
                 kernel_size,
                 demodulate=True,
                 up=1,
                 down=1,
                 padding=0
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.up = up
        self.down = down
        self.padding = padding

        # Scaling factor for weight init
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        # Learnable conv weights
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        # Linear layer for style modulation
        self.modulation = nn.Linear(style_dim, in_channels, bias=False)

    def forward(self, x, style):
        """
        Forward pass of modulated convolution layer.
        
        Args:
            x (Tensor): Input feature map.
            style (Tensor): Style vector.
            noise (Tensor, optional): Noise tensor for injection.
        
        Returns:
            Tensor: Output feature map after modulated convolution and noise injection.
        """
        batch, in_channels, height, width = x.shape

        # Style Modulation
        # Transform style vector to match input channels
        style = self.modulation(style).view(batch, in_channels, 1, 1)
        # Scale weights and multiply by style
        weight = self.scale * self.weight.unsqueeze(0) * style.unsqueeze(1)

        # Demodulation
        if self.demodulate:
            # Calc demodulation factor
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            # Apply demodulation to weights
            weight = weight * demod.view(batch, self.out_channels, 1, 1, 1)

        # Reshape weight for conv
        # Combine batch and out_channels dimensions
        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )

        # Stride and padding for up/downsampling
        stride = (1 / self.up) if self.up > 1 else self.down
        if isinstance(stride, float):
            stride = int(1 / stride)
        padding = self.kernel_size // 2 if self.padding == 0 else self.padding

        # Perform conv
        # Reshape input to combine batch and in_channels
        x = x.reshape(1, batch * in_channels, height, width)
        # Apply grouped conv (1 group per batch item)
        out = F.conv2d(x, weight, padding=padding, stride=stride, groups=batch)
        # Reshape output to original batch size
        out = out.view(batch, self.out_channels, out.size(2), out.size(3))

        return out


class SynthesisBlock(nn.Module):
    """
    Synthesis Block for StyleGAN2.

    Block contains modulated convolution, noise injection, and activation.

    Args:
        in_channels (int): Num input channels.
        out_channels (int): Num output channels.
        style_dim (int): Dim of style vector.
        kernel_size (int, optional): Size of conv kernel. Defaults to 3.
        up (int, optional): Upsampling factor. Defaults to 1.
        final_resolution (tuple, optional): Final resolution for last block. Defaults to None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 style_dim,
                 kernel_size=3,
                 up=1,
                 final_resolution=None
                ):
        super().__init__()
        self.up = up
        self.final_resolution = final_resolution
        self.conv1 = ModulatedConv2d(in_channels, out_channels, style_dim, kernel_size=kernel_size)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, style_dim, kernel_size=kernel_size, up=up)
        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x, style, noise=None):
        """Forward pass of the Synthesis block."""
        # Decerease channel num first - transform features first, then upsample
        x = self.conv1(x, style)
        x = self.noise1(x, noise=noise)
        x = self.activate(x)
        # Upsampling handled in ModulatedConv2d
        x = self.conv2(x, style)  
        x = self.noise2(x, noise=noise)
        x = self.activate(x)
        
        if self.final_resolution: # Get 256x240 output via this
            x = F.interpolate(x, size=self.final_resolution, mode='bilinear', align_corners=False)
        return x


class StyleGAN2Generator(nn.Module):
    """
    Generator class for StyleGAN2.

    Generates grayscale images from latent vectors - size 256x240.
    Mapping network transforms input latent code.
    Synthesis network generates images. Uses progressive growing 
    through strided modulated convolutions.

    Args:
        z_dim (int): Dim of input latent vector.
        w_dim (int): Dim of intermediate latent space.
        num_mapping_layers (int): Num layers in mapping network.
        mapping_dropout (float): Dropout rate for mapping network.
        label_dim (int): Num classes for conditional generation. Defaults to 2 (AD, NC).
        num_layers (int, optional): Num layers in synthesis network. Defaults to 5.
        ngf (int, optional): Num generator features. Defaults to 64.
    """
    def __init__(self, 
                 z_dim,
                 w_dim,
                 num_mapping_layers,
                 mapping_dropout,
                 label_dim = 2, 
                 num_layers = 5,
                 ngf = 64
                ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_mapping_layers = num_mapping_layers
        self.label_dim = label_dim
        self.num_layers = num_layers
        self.ngf = ngf
        self.mapping_dropout = mapping_dropout
        
        # Init mapping network
        self.mapping_network = MappingNetwork(self.z_dim,
                                              self.w_dim,
                                              self.num_mapping_layers,
                                              self.label_dim,
                                              self.mapping_dropout)
        
        # Init synthesis network
        self.synthesis_network = nn.ModuleList()
        
        # Learnable constant input (8x8)
        self.const = nn.Parameter(torch.randn(1, ngf * 8, 8, 8))
        
        # Create synthesis architecture
        # Use strided modulated convolutions for progressive growing
        # Progression: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x240
        channel_multipliers = [8, 4, 2, 1, 1/2]
        in_channels = self.ngf * 8  # Start with the number of channels in const
        for i in range(num_layers):
            out_channels = int(self.ngf * channel_multipliers[i])
            up = 2
            final_resolution = None
            if i == num_layers - 1:  # Last layer
                final_resolution = (256, 256)  # Use interpolation for final size
            self.synthesis_network.append(
                SynthesisBlock(in_channels, out_channels, w_dim, up=up, final_resolution=final_resolution)
            )
            in_channels = out_channels
            
        # Add a final convolution layer to reduce to 1 channel
        self.to_rgb = nn.Conv2d(int(self.ngf * channel_multipliers[-1]), 1, kernel_size=1)



    def forward(self, z, labels, return_latents=False):
        """
        Forward pass of the StyleGAN2 generator.

        Args:
            z (torch.Tensor): Input latent vector.
            labels (torch.Tensor): Class labels for conditional generation.
            return_latents (bool): If True, return both generated images and w latents.

        Returns:
            torch.Tensor: Generated grayscale image of size 256x240,
                        and w latents if return_latents is True.
        """
        batch_size = z.shape[0]
        # Generate w from z
        w = self.mapping_network(z, labels)
        # Start with learned constant - repeated for batch
        x = self.const.repeat(batch_size, 1, 1, 1)
        # Apply synthesis blocks
        for block in self.synthesis_network:
            x = block(x, w)
        
        # Apply the final convolution to get 1 channel output
        x = self.to_rgb(x)
        x = torch.tanh(x)  # Force output range [-1, 1]
        
        if return_latents:
            return x, w
        return x
    
    
class ResidualBlock(nn.Module):
    """
    Residual blcok for StyleGAN2.

    2 convs, skip connection, optional downsampling.
    Args:
        in_channels (int): Num input channels
        out_channels (int): Num output channels
        downsample (bool, optional): Use avg pooling to downsample? Defaults to True.
    """
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.downsample = downsample
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass of Residual Block.

        Args:
            x (torch.Tensor): Input tensor - shape [N, in_channels, H, W]

        Returns:
            torch.Tensor: Processed tensor - shape [N, out_channels, H', W']
        """
        residual = self.skip(x)
        if self.downsample:
            residual = F.avg_pool2d(residual, 2)

        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        if self.downsample:
            out = F.avg_pool2d(out, 2)

        return out + residual
    
    
class MiniBatchStdDev(nn.Module):
    """
    Caclulate std dev of features across mini-batches, add to input features.
    Helps discriminator better find real and generated images.
    
    Args:
            group_size (int, optional): Num images per group for std calc. Defaults to 4.
                              If None - use whole batch.
            num_new_features (int, optional): Num stddev features to add. Defaults to 1.
    """
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        """
        Forward pass of MiniBatchStdDev layer.

        Args:
            x (torch.Tensor): Input tensor - shape [N, C, H, W]

        Returns:
            torch.Tensor: Output tensor with stddev features - shape [N, C+num_new_features, H, W]
        """
        N, C, H, W = x.shape
        
        # Calc group size
        G = min(self.group_size or N, N) # Determine group size, default to N
        F = self.num_new_features
        c = C // F # Number of channels per feature
        
        # Reshape input: [G, M, F, c, H, W]
        # G: num groups, M: num images per group (N/G), F: num new features, c: channels per feature
        y = x.reshape(G, -1, F, c, H, W)
        # Minus group mean
        y = y - y.mean(dim=0)
        # Calc variance of group
        y = y.square().mean(dim=0) # [M, F, c, H, W]
        # Calc stddev for each group
        # Add epsilon (1e-8) so no div by zero
        y = (y + 1e-8).sqrt()           # [M, F, c, H, W]
        # Avg over channels and spatial dims
        y = y.mean(dim=[2,3,4])         # [M, F]
        # Reshape - add singular dims
        y = y.reshape(-1, F, 1, 1)      # [M, F, 1, 1]
        # Match input spatial dims
        y = y.repeat(G, 1, H, W)        # [N, F, H, W]
        # Concat calcd stddev to original input
        output = torch.cat([x, y], dim=1)    # [N, C+F, H, W]

        return output
    
    
class StyleGAN2Discriminator(nn.Module):
    """
    Discriminator for StyleGAN2.

    Args:
        image_size (tuple): Size of input image (H, W).
        num_channels (int): Num input channels.
        ndf (int): Num discriminator features.
        num_layers (int): Num downsampling layers.
    """
    def __init__(self, image_size, num_channels, ndf, num_layers):
        super().__init__()
        self.image_size = image_size  # (256, 256)
        self.num_channels = num_channels  # 1
        self.ndf = ndf  # 64
        self.num_layers = num_layers  # 5
        
        self.discrim_network = nn.ModuleList()
        
        # Initial conv layer
        # In: [batch_size, 1, 256, 256]
        # Out: [batch_size, 64, 256, 256]
        self.discrim_network.append(nn.Conv2d(in_channels=num_channels, out_channels=ndf, kernel_size=3, padding=1))
        
        # Residual blocks
        # Set up channel progression
        in_channels = ndf
        channel_multipliers = [1, 2, 4, 8, 16]  # Progression of channel multiplication
        for i in range(num_layers):
            out_channels = int(ndf * channel_multipliers[i])
            # In: [batch_size, in_channels, H, W]
            # Out: [batch_size, out_channels, H/2, W/2]
            self.discrim_network.append(ResidualBlock(in_channels, out_channels, downsample=True))
            in_channels = out_channels
            
        # Spatial dims after downsamples
        # 5 downsamples: 256/(2^5) = 8
        final_height = image_size[0] // (2 ** num_layers)  # 8
        final_width = image_size[1] // (2 ** num_layers)   # 8
            
        # Add MiniBatchStdDev layer (adds 1 to channel dim)
        # In: [batch_size, 1024 (64 * 16), 8, 8]
        # Out: [batch_size, 1025, 8, 8]
        self.minibatch_stddev = MiniBatchStdDev()
        
        # Final conv layer
        self.final_conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1)
        
        # Flattening layer
        # In: [batch_size, 1024, 8, 8]
        # Out: [batch_size, 1024 * 8 * 8] = 65536
        self.flatten = nn.Flatten()
        
        # Dense layer - classifier
        out_features = in_channels * final_height * final_width
        # In: [batch_size, 65536]
        # Out: [batch_size, 1]
        self.final_linear = nn.Linear(out_features, 1)

    def forward(self, x, feature_output=False):
        """
        Forward pass of StyleGAN2 discriminator.

        Args:
            x (torch.Tensor): Input tensor - shape [batch_size, num_channels, height, width].
            feature_output (bool): If True, return features instead of final output.

        Returns:
            torch.Tensor: Discriminator scores of shape (batch_size,) or features if feature_output is True.
        """
        print(f"Input shape: {x.shape}")
        features = []
        for i, layer in enumerate(self.discrim_network):
            x = layer(x)
            print(f"After layer {i}: {x.shape}")
            features.append(x)
        
        x = self.minibatch_stddev(x)
        print(f"After MiniBatchStdDev: {x.shape}")
        x = self.final_conv(x)
        print(f"After final conv: {x.shape}")
        features.append(x)
        
        x = self.flatten(x)
        print(f"After flatten: {x.shape}")
        final_features = x  # Store the flattened features
        x = self.final_linear(x)
        print(f"After final linear: {x.shape}")

        # Get single value per sample
        x = x.squeeze(1)

        if feature_output:
            return final_features
        else:
            return x