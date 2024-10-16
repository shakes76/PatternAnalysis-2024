import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class EQLinearLayer(nn.Module):
    """
    Fully Connected Layer - Equalised Learning Rate
    """
    def __init__(self, in_dim, out_dim, bias=0.) -> None:
        super(EQLinearLayer, self).__init__()
        self.scale = 1 / math.sqrt(np.prod(in_dim))
        self.weight = nn.Parameter(torch.randn([out_dim, in_dim])) * self.scale
        self.bias = nn.Parameter(torch.ones(out_dim) * bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class FCBlock(nn.Module):
    """
    Fully Connected Noise Mapping Network.
    """
    def __init__(self, z_dim, w_dim) -> None:
        super(FCBlock, self).__init__()
        self.net = nn.Sequential(
            EQLinearLayer(z_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
            EQLinearLayer(w_dim, w_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # Pixel-wise normalisation for input
        x = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return self.net(x)

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalisation
    """
    def __init__(self, channels, w_dim) -> None:
        super(AdaIN, self).__init__()
        self.inst_norm = nn.InstanceNorm2d(channels)
        self.style_weight = EQLinearLayer(w_dim, channels) # scale of style
        self.style_bias = EQLinearLayer(w_dim, channels) # style shift

    def forward(self, x, w):
        # Instance Normalisation
        x = self.inst_norm(x)

        # Apply style transformation
        style_weight = self.style_weight(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)

        # Apply style to normalized input
        return x * style_weight + style_bias

class NoiseInjection(nn.Module):
    """
    Inject noise into the synthesis network.
    """
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise

class StyleBlock(nn.Module):
    """
    Single Style Block For Synthesis Network
    """
    def __init__(self, in_channels, out_channels, w_dim, upsample=False) -> None:
        super(StyleBlock, self).__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise_inj = NoiseInjection(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.ada_in = AdaIN(out_channels, w_dim)

    def forward(self, x, w, noise=None):
        if self.upsample:
            # Up-sample image
            x = F.interpolate(x, scale_factor=2,
                mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.noise_inj(x, noise)
        x = self.activation(x)
        x = self.ada_in(x, w)
        return x

class GeneratorBlock(nn.Module):
    """
    Generator Block - Comprised of Multiple Style Blocks
    """
    def __init__(self, in_channels, out_channels, w_dim, upsample=True) -> None:
        super(GeneratorBlock, self).__init__()

        # Style blocks
        self.block1 = StyleBlock(in_channels, out_channels, w_dim, upsample)
        self.block2 = StyleBlock(out_channels, out_channels, w_dim, upsample=False)

        # Output of last layer is converted to RGB
        self.to_rgb_weight = nn.Parameter(torch.randn(3, out_channels, 1, 1))
        self.to_rgb_bias = nn.Parameter(torch.zeros(3))
        self.to_rgb_scale = 1 / math.sqrt(out_channels)

    def forward(self, x, w, noise):
        # Apply style blocks
        x = self.block1(x, w, noise[0])
        x = self.block2(x, w, noise[1])

        # Convert to RGB using F.conv2d
        rgb = F.conv2d(x, self.to_rgb_weight * self.to_rgb_scale, self.to_rgb_bias)

        return x, rgb

class Generator(nn.Module):
    """
    Generator Network - StyleGAN1.
    """
    def __init__(self, z_dim, w_dim, in_channels, num_layers) -> None:
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.in_channels = in_channels
        self.num_layers = num_layers

        # Mapping network
        self.mapping = FCBlock(z_dim, w_dim)

        # Initial constant input
        self.const_in = nn.Parameter(
            torch.randn(1, in_channels, 4, 4)
        )

        # Generator block for constant input
        self.const_noise_inj = NoiseInjection(in_channels)
        self.const_activation = nn.LeakyReLU(0.2)
        self.const_ada_in = AdaIN(in_channels, w_dim)
        self.const_style_block = StyleBlock(in_channels, in_channels, w_dim)

        # Generator blocks
        self.gen_blocks = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_layers - 1):  # One less generator block
            out_channels = max(
                in_channels // (2 ** (i + 1)), 16 # Minimum 16 channels
            )
            self.gen_blocks.append(
                GeneratorBlock(current_channels, out_channels, w_dim)
            )
            current_channels = out_channels

    def forward(self, z, truncation_psi=0.7, truncation_cutoff=8):
        batch_size = z.shape[0]

        # Map z to intermediate space W
        w = self.mapping(z)
        # Apply truncation trick
        if truncation_psi < 1:
            w_avg = torch.zeros_like(w).mean(dim=0, keepdim=True)
            layer_indices = torch.arange(self.num_layers + 1)[None, :, None].to(z.device)
            cutoff = torch.ones_like(layer_indices, dtype=torch.float32) * truncation_cutoff
            mask = (layer_indices < cutoff).float()
            w = w_avg + (w - w_avg) * (mask * truncation_psi + (1 - mask))

        # Start with constant input
        x = self.const_in.repeat(batch_size, 1, 1, 1)

        # Generate noise for each layer
        noise = [
            torch.randn(
                batch_size, 1, 4 * 2**i, 4 * 2**i, device=z.device
            ) for i in range(self.num_layers)
        ] * 2

        # Apply initial transformations
        x = self.const_noise_inj(x, noise[0])
        x = self.const_activation(x)
        x = self.const_ada_in(x, w[:, 0])
        x = self.const_style_block(x, w[:, 1], noise[1])

        # Apply generator blocks
        rgb_outs = []
        for i, block in enumerate(self.gen_blocks):
            x, rgb = block(x, w[:, i+2], [noise[2*i+2], noise[2*i+3]])
            rgb_outs.append(rgb)

        # Combine RGB outputs
        image = rgb_outs[-1]
        for rgb in reversed(rgb_outs[:-1]):
            image = F.interpolate(image, scale_factor=2, mode='bilinear', align_corners=False)
            image = image + rgb

        return image

    def generate_truncated(self, num_samples,
        truncation_psi=0.7, truncation_cutoff=8):
            latent_z = torch.randn(
                num_samples, self.z_dim, device=self.const_in.device
            )
            return self.forward(latent_z, truncation_psi, truncation_cutoff)

    def get_latent(self, z):
        return self.mapping(z)

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.z_dim, device=self.const_in.device
        )
        latent = self.get_latent(latent_in).mean(0, keepdim=True)
        return latent

class EQConv2d(nn.Module):
    """
    Conv2d with Equalised Learning Rate
    """
    def __init__(self, in_channels, out_channels, kernel_size,
        stride=1, padding=0) -> None:
        super(EQConv2d, self).__init__()
        self.scale = 1 / math.sqrt(in_channels * kernel_size ** 2)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(
            x, self.weight * self.scale, self.bias,
            stride=self.stride, padding=self.padding
        )

class DiscriminatorBlock(nn.Module):
    """
    Discriminator Block - ProGAN
    """
    def __init__(self, in_channels, out_channels, downsample=True) -> None:
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = EQConv2d(in_channels,
            in_channels, 3, padding=1)
        self.conv2 = EQConv2d(in_channels,
            out_channels, 3, padding=1)
        self.downsample = downsample

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        if self.downsample:
            x = F.avg_pool2d(x, 2)

        return x

class MinibatchStdDev(nn.Module):
    """
    Minibatch Standard Deviation Layer
    """
    def __init__(self):
        super(MinibatchStdDev, self).__init__()

    def forward(self, x):
        batch_size, _, height, width = x.shape
        std = torch.std(x, dim=0, unbiased=False)
        mean_std = torch.mean(std)
        mean_std = mean_std.expand((batch_size, 1, height, width))
        return torch.cat([x, mean_std], dim=1)

class Discriminator(nn.Module):
    """
    Discriminator Network - ProGAN
    """
    def __init__(self, image_size, channels_base=32, max_channels=512) -> None:
        super(Discriminator, self).__init__()
        self.image_size = image_size
        num_layers = int(math.log2(image_size)) - 1

        # Layer to convert image from RGB
        self.from_rgb = nn.Sequential(
            EQConv2d(3, channels_base, 1),
            nn.LeakyReLU(0.2),
        )

        # Discriminator Blocks
        self.blocks = nn.ModuleList()
        in_channels = channels_base
        for i in range(num_layers):
            out_channels = min(in_channels * 2, max_channels)
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels))
            in_channels = out_channels

        # Final layers
        self.minibatch_stddev = MinibatchStdDev()
        self.conv = EQConv2d(in_channels + 1, in_channels, 3, padding=1)
        self.fc = EQLinearLayer(in_channels * 4 * 4, in_channels)
        self.output = EQLinearLayer(in_channels, 1)


    def forward(self, x):
        x = self.from_rgb(x)

        for block in self.blocks:
            x = block(x)

        x = self.minibatch_stddev(x)
        x = F.leaky_relu(self.conv(x), 0.2)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc(x), 0.2)
        x = self.output(x)

        return x
