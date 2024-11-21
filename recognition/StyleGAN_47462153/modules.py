import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelNorm(nn.Module):
    """
    Normalizes pixel values across channels to stabilize training.
    
    Attributes:
        epsilon (float): Small constant for numerical stability.
    """
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        """
        Forward pass for pixel normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer that applies learned style transformations.
    
    Attributes:
        instance_norm (nn.InstanceNorm2d): Instance normalization layer.
        style_scale (WSLinear): Linear layer for style scaling.
        style_bias (WSLinear): Linear layer for style biasing.
    """
    def __init__(self, channels, w_dim):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale = WSLinear(w_dim, channels)
        self.style_bias = WSLinear(w_dim, channels)

    def forward(self, x, w):
        """
        Applies instance normalization followed by learned style scaling and biasing.
        
        Args:
            x (torch.Tensor): Input feature map.
            w (torch.Tensor): Intermediate latent vector.
        
        Returns:
            torch.Tensor: Styled and normalized feature map.
        """
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias

class WSLinear(nn.Module):
    """
    Linear layer with weight scaling and custom initialization for stability.
    
    Attributes:
        linear (nn.Linear): Linear layer for feature transformation.
        scale (float): Scaling factor for weight normalization.
        bias (torch.Tensor): Bias tensor for the layer.
    """
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5  
        self.bias = self.linear.bias
        self.linear.bias = None
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.bias)  

    def forward(self, x):
        """
        Forward pass with scaled weights.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Scaled linear transformation.
        """
        return self.linear(x * self.scale) + self.bias

class WSConv2d(nn.Module):
    """
    Convolutional layer with weight scaling and custom initialization.
    
    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        scale (float): Scaling factor for weight normalization.
        bias (torch.Tensor): Bias tensor for the layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * kernel_size ** 2)) ** 0.5  
        self.bias = self.conv.bias
        self.conv.bias = None
        nn.init.normal_(self.conv.weight, mean=0.0, std=1.0)  
        nn.init.zeros_(self.bias)  

    def forward(self, x):
        """
        Forward pass with scaled weights.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Convolved tensor with scaled weights.
        """
        return self.conv(x * self.scale) + self.bias.view(1, -1, 1, 1)

class InjectNoise(nn.Module):
    """
    Injects random noise into the input feature map with learnable scaling.
    
    Attributes:
        weight (torch.nn.Parameter): Learnable scaling factor for noise.
    """
    def __init__(self, channels):
        super(InjectNoise, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))  

    def forward(self, x):
        """
        Adds scaled random noise to the input.
        
        Args:
            x (torch.Tensor): Input feature map.
        
        Returns:
            torch.Tensor: Feature map with added noise.
        """
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        return x + self.weight * noise

class GenBlock(nn.Module):
    """
    Generator block with AdaIN, noise injection, and convolutional layers.
    
    Attributes:
        conv1, conv2 (WSConv2d): Convolutional layers.
        inject_noise1, inject_noise2 (InjectNoise): Noise injection layers.
        adain1, adain2 (AdaIN): Adaptive Instance Normalization layers.
        leaky (nn.LeakyReLU): Activation function.
    """
    def __init__(self, in_channels, out_channels, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)
        self.conv2 = WSConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.inject_noise1 = InjectNoise(out_channels)
        self.inject_noise2 = InjectNoise(out_channels)
        self.adain1 = AdaIN(out_channels, w_dim)
        self.adain2 = AdaIN(out_channels, w_dim)

    def forward(self, x, w):
        """
        Forward pass with noise injection, convolution, and AdaIN.
        
        Args:
            x (torch.Tensor): Input feature map.
            w (torch.Tensor): Intermediate latent vector.
        
        Returns:
            torch.Tensor: Processed feature map.
        """
        x = self.conv1(x)
        x = self.inject_noise1(x)
        x = self.leaky(x)
        x = self.adain1(x, w)
        x = self.conv2(x)
        x = self.inject_noise2(x)
        x = self.leaky(x)
        x = self.adain2(x, w)
        return x

class MappingNetwork(nn.Module):
    """
    Maps latent vector z to an intermediate latent vector w for controlling style.
    
    Attributes:
        z_dim (int): Dimension of the input latent space (z).
        w_dim (int): Dimension of the intermediate latent space (w).
        mapping (nn.Sequential): Sequential model of fully connected layers 
                                 and activations for mapping z to w.
    """
    def __init__(self, z_dim, w_dim):
        super(MappingNetwork, self).__init__()
        layers = [PixelNorm()]  # Normalize input to stabilize training
        for _ in range(8):  # 8 fully connected layers with LeakyReLU as in StyleGAN
            layers.append(WSLinear(z_dim if _ == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the mapping network, transforming latent vector z to w.
        
        Args:
            x (torch.Tensor): Input latent vector z.
        
        Returns:
            torch.Tensor: Intermediate latent vector w.
        """
        return self.mapping(x)

class Generator(nn.Module):
    """
    Generator network for synthesizing images.
    
    Attributes:
        initial_constant (torch.nn.Parameter): Learnable constant input.
        initial_noise (InjectNoise): Initial noise layer.
        initial_conv (WSConv2d): Initial convolution layer.
        initial_adain (AdaIN): Initial AdaIN layer.
        progression_blocks (nn.ModuleList): List of GenBlocks for progressive synthesis.
        to_rgb (WSConv2d): Converts features to image channels.
    """
    def __init__(self, z_dim=512, w_dim=512, in_channels=512, img_channels=1):
        super(Generator, self).__init__()
        self.initial_constant = nn.Parameter(torch.ones(1, in_channels, 4, 4))  
        self.initial_noise = InjectNoise(in_channels)
        self.initial_conv = WSConv2d(in_channels, in_channels)
        self.leaky = nn.LeakyReLU(0.2)
        self.mapping = MappingNetwork(z_dim, w_dim)
        self.initial_adain = AdaIN(in_channels, w_dim)

        self.progression_blocks = nn.ModuleList([
            GenBlock(in_channels, in_channels, w_dim),  
            GenBlock(in_channels, in_channels, w_dim),  
            GenBlock(in_channels, in_channels, w_dim),  
            GenBlock(in_channels, in_channels, w_dim),  
        ])

        self.to_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, padding=0)  

    def forward(self, noise):
        """
        Generates an image from a latent vector.
        
        Args:
            noise (torch.Tensor): Input latent vector z.
        
        Returns:
            torch.Tensor: Generated image.
        """
        w = self.mapping(noise)
        x = self.initial_constant.repeat(noise.size(0), 1, 1, 1)  
        x = self.initial_noise(x)
        x = self.leaky(self.initial_conv(x))
        x = self.initial_adain(x, w)

        for block in self.progression_blocks:
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')  
            x = block(x, w)  

        out = self.to_rgb(x)  
        return out

class ConvBlock(nn.Module):
    """
    Convolutional block for discriminator with weight scaling and LeakyReLU.
    
    Attributes:
        conv1, conv2 (WSConv2d): Convolutional layers.
        leaky (nn.LeakyReLU): Activation function.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, in_channels)
        self.conv2 = WSConv2d(in_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass through two convolutional layers.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Processed tensor.
        """
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x

class Discriminator(nn.Module):
    """
    Discriminator network for real/fake image classification.
    
    Attributes:
        from_rgb (WSConv2d): Converts image to feature map.
        progression_blocks (nn.ModuleList): List of ConvBlocks for progressive downsampling.
        final_conv (WSConv2d): Final convolution layer.
        linear (WSLinear): Linear layer for real/fake prediction.
    """
    def __init__(self, in_channels=512, img_channels=1):
        super(Discriminator, self).__init__()
        self.from_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, padding=0)  
        self.progression_blocks = nn.ModuleList([
            ConvBlock(in_channels, in_channels),  
            ConvBlock(in_channels, in_channels),  
            ConvBlock(in_channels, in_channels),  
            ConvBlock(in_channels, in_channels),  
        ])

        self.leaky = nn.LeakyReLU(0.2)
        self.final_conv = WSConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.linear = WSLinear(in_channels * 4 * 4, 1)  

    def forward(self, x):
        """
        Forward pass through the discriminator network.
        
        Args:
            x (torch.Tensor): Input image tensor.
        
        Returns:
            torch.Tensor: Real/fake prediction score.
        """
        x = self.from_rgb(x)  
        for block in self.progression_blocks:
            x = block(x) 
            x = nn.functional.avg_pool2d(x, 2) 

        x = self.leaky(self.final_conv(x))  
        x = x.view(x.size(0), -1)  
        x = self.linear(x)
        return x
