import torch 
from torch import nn
import torch.nn.functional as F

class WSLinear(nn.Module):
    """Weighted Sum Linear Layer with Learned Scale."""
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # Linear layer
        self.scale = (2 / in_features) ** 0.5  # Scale factor for normalization
        self.bias = self.linear.bias  # Retrieve bias
        self.linear.bias = None  # Disable the bias in the linear layer

        # Initialize weights and bias
        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """Forward pass with scaling."""
        return self.linear(x * self.scale) + self.bias

class PixenNorm(nn.Module):
    """Pixel Normalization Layer."""
    def __init__(self):
        super(PixenNorm, self).__init__()
        self.epsilon = 1e-8  # Small constant for numerical stability

    def forward(self, x):
        """Normalize input tensor across channels."""
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)

class MappingNetwork(nn.Module):
    """Mapping Network for transforming latent vectors."""
    def __init__(self, z_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            PixenNorm(),
            WSLinear(z_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
            nn.ReLU(),
            WSLinear(w_dim, w_dim),
        )

    def forward(self, x):
        """Map latent space to style space."""
        return self.mapping(x)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization Layer."""
    def __init__(self, channels, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)  # Instance normalization
        self.style_scale = WSLinear(w_dim, channels)  # Scale parameter
        self.style_bias = WSLinear(w_dim, channels)  # Bias parameter

    def forward(self, x, w):
        """Apply AdaIN to the input."""
        x = self.instance_norm(x)  # Normalize input
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)  # Reshape for scaling
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)  # Reshape for bias
        return style_scale * x + style_bias  # Scale and shift

class injectNoise(nn.Module):
    """Layer to inject noise into the input tensor."""
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))  # Learnable noise weight

    def forward(self, x):
        """Inject noise to the input."""
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)  # Generate noise
        return x + self.weight + noise  # Add noise to input

class GenBlock(nn.Module):
    """Generator Block for building the generator architecture."""
    def __init__(self, in_channel, out_channel, w_dim):
        super(GenBlock, self).__init__()
        self.conv1 = WSConv2d(in_channel, out_channel)  # First convolution
        self.conv2 = WSConv2d(out_channel, out_channel)  # Second convolution
        self.leaky = nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU activation
        self.inject_noise1 = injectNoise(out_channel)  # Noise injection layer
        self.inject_noise2 = injectNoise(out_channel)  # Another noise injection layer
        self.adain1 = AdaIN(out_channel, w_dim)  # First AdaIN layer
        self.adain2 = AdaIN(out_channel, w_dim)  # Second AdaIN layer

    def forward(self, x, w):
        """Forward pass through the generator block."""
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)  # Pass through layers
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)  # Pass through layers
        return x

# Factors for adjusting channels at different levels
factors = [1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

class Generator(nn.Module):
    """Generator model for the GAN architecture."""
    def __init__(self, z_dim, w_dim, in_channels, img_channels=3):
        super().__init__()
        self.starting_cte = nn.Parameter(torch.ones(1, in_channels, 4, 4))  # Initial constant tensor
        self.map = MappingNetwork(z_dim, w_dim)  # Mapping network
        self.initial_adain1 = AdaIN(in_channels, w_dim)  # First AdaIN layer
        self.initial_adain2 = AdaIN(in_channels, w_dim)  # Second AdaIN layer
        self.initial_noise1 = injectNoise(in_channels)  # First noise injection
        self.initial_noise2 = injectNoise(in_channels)  # Second noise injection
        self.initial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)  # Initial convolution
        self.leaky = nn.LeakyReLU(0.2, inplace=True)  # Leaky ReLU activation

        self.initial_rgb = WSConv2d(in_channels, img_channels, kernel_size=1, stride=1, padding=0)  # Initial RGB layer
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb])  # Module lists for blocks and RGB layers

        # Create progressive blocks and RGB layers
        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])  # Input channels for current block
            conv_out_c = int(in_channels * factors[i + 1])  # Output channels for current block
            self.prog_blocks.append(GenBlock(conv_in_c, conv_out_c, w_dim))  # Add generator block
            self.rgb_layers.append(WSConv2d(conv_out_c, img_channels, kernel_size=1, stride=1, padding=0))  # Add RGB layer

    def fade_in(self, alpha, upscaled, generated):
        """Fade in between two images."""
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, noise, alpha, steps):
        """Forward pass through the generator."""
        w = self.map(noise)  # Map latent noise to style
        x = self.initial_adain1(self.initial_noise1(self.starting_cte), w)  # Initial processing
        x = self.initial_conv(x)  # Initial convolution
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)  # Further processing

        if steps == 0:
            return self.initial_rgb(x)  # Return initial RGB output

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode='bilinear')  # Upscale previous output
            out = self.prog_blocks[step](upscaled, w)  # Process through generator block

        final_upscaled = self.rgb_layers[steps - 1](upscaled)  # Final RGB processing for upscaled output
        final_out = self.rgb_layers[steps](out)  # Final RGB processing for current output

        return self.fade_in(alpha, final_upscaled, final_out)  # Combine outputs with fade-in

class WSConv2d(nn.Module):
    """Weighted Sum Convolutional Layer."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)  # Convolutional layer
        self.scale = (2 / (in_channels * (kernel_size ** 2))) ** 0.5  # Scale factor for normalization
        self.bias = self.conv.bias  # Retrieve bias
        self.conv.bias = None  # Disable bias in the convolutional layer

        # Initialize weights and bias
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """Forward pass with scaling."""
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)  # Apply convolution and add bias

class ConvBlock(nn.Module):
    """Convolutional Block for Discriminator."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2d(in_channels, out_channels)  # First convolution
        self.conv2 = WSConv2d(out_channels, out_channels)  # Second convolution
        self.leaky = nn.LeakyReLU(0.2)  # Leaky ReLU activation

    def forward(self, x):
        """Forward pass through the convolutional block."""
        x = self.leaky(self.conv1(x))  # Apply first convolution and activation
        x = self.leaky(self.conv2(x))  # Apply second convolution and activation
        return x

class Discriminator(nn.Module):
    """Discriminator model for the GAN architecture."""
    def __init__(self, in_channels, img_channels=3):
        super(Discriminator, self).__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])  # Module lists for blocks and RGB layers
        self.leaky = nn.LeakyReLU(0.2)  # Leaky ReLU activation

        # Create progressive blocks and RGB layers
        for i in range(len(factors) - 1, 0, -1):
            conv_in = int(in_channels * factors[i])  # Input channels for current block
            conv_out = int(in_channels * factors[i - 1])  # Output channels for current block
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))  # Add convolutional block
            self.rgb_layers.append(
                WSConv2d(img_channels, conv_in, kernel_size=1, stride=1, padding=0)  # Add RGB layer
            )

        self.initial_rgb = WSConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )  # Initial RGB layer
        self.rgb_layers.append(self.initial_rgb)  # Append to RGB layers
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # Average pooling layer

        self.final_block = nn.Sequential(  # Final sequential block for output
            WSConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha, downscaled, out):
        """Fade in between two images."""
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        """Calculate the minibatch standard deviation."""
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)  # Concatenate to input

    def forward(self, x, alpha, steps):
        """Forward pass through the discriminator."""
        cur_step = len(self.prog_blocks) - steps  # Current step index
        out = self.leaky(self.rgb_layers[cur_step](x))  # Initial processing

        if steps == 0:
            out = self.minibatch_std(out)  # Add minibatch std if at step 0
            return self.final_block(out).view(out.shape[0], -1)  # Final output processing

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))  # Downscale previous output
        out = self.avg_pool(self.prog_blocks[cur_step](out))  # Process through current block
        out = self.fade_in(alpha, downscaled, out)  # Fade in between downscaled and current output

        # Process through remaining blocks
        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)  # Calculate minibatch std for final output
        return self.final_block(out).view(out.shape[0], -1)  # Final output processing
