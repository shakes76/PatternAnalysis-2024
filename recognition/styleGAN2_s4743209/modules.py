import torch
import torch.nn as nn
import torch.nn.functional as F


class AAConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Style modulation
        self.style_scale = nn.Linear(style_dim, out_channels)
        self.style_bias = nn.Linear(style_dim, out_channels)

        # Self-attention components
        self.query = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.key = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.value = nn.Conv2d(out_channels, out_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

        self._init_weights()

    def _init_weights(self):
        for m in [self.conv, self.query, self.key, self.value]:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

        # Initialize style modulation layers
        for m in [self.style_scale, self.style_bias]:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def attention(self, x):
        batch, C, H, W = x.size()

        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H * W)
        value = self.value(x).view(batch, -1, H * W)

        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)

        return self.gamma * out + x

    def forward(self, x, style):
        x = self.conv(x)

        # Apply style modulation
        scale = self.style_scale(style).view(style.size(0), -1, 1, 1)
        bias = self.style_bias(style).view(style.size(0), -1, 1, 1)
        x = x * (1 + scale) + bias

        x = self.attention(x)
        x = self.norm(x)
        return self.activation(x)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        layers = []
        in_dim = z_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])
            in_dim = w_dim

        self.mapping = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        return self.mapping(z)


class Generator(nn.Module):
    def __init__(self, style_dim, num_layers=7, channels=(512, 512, 512, 512, 256, 128, 64), img_size=256):
        super().__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        self.channels = channels
        self.img_size = img_size

        # Constant input
        self.const = nn.Parameter(torch.randn(1, channels[0], 4, 4))

        # AA Conv layers
        self.aa_layers = nn.ModuleList()
        in_chan = channels[0]
        for out_chan in channels:
            self.aa_layers.append(AAConvLayer(in_chan, out_chan, style_dim))
            in_chan = out_chan

        # To RGB layer
        self.to_rgb = nn.Conv2d(channels[-1], 1, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.to_rgb.weight.data, 0.0, 0.02)
        nn.init.constant_(self.to_rgb.bias.data, 0)

    def forward(self, styles):
        batch_size = styles.size(0)

        # Start with constant input
        x = self.const.repeat(batch_size, 1, 1, 1)

        # Process through AA-Conv layers
        for i, aa_layer in enumerate(self.aa_layers):
            # Get style for current layer
            if styles.dim() == 3:
                style = styles[:, i]
            else:
                style = styles

            x = aa_layer(x, style)

            # Upsample after each layer except the last
            if i != len(self.aa_layers) - 1:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Generate final image
        x = self.to_rgb(x)

        if x.size(-1) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 512, 512, 512)):
        super().__init__()
        self.channels = channels

        layers = []
        in_channel = 1  # Grayscale input
        for out_channel in channels:
            layers.extend([
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_channel),
                nn.LeakyReLU(0.2)
            ])
            in_channel = out_channel

        self.main = nn.Sequential(*layers)
        self.final = nn.Conv2d(channels[-1], 1, 4, padding=0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.main(x)
        return self.final(x).view(x.size(0), -1)


class StyleGAN2(nn.Module):
    def __init__(self, w_dim, num_layers, channels, img_size=256):
        super().__init__()
        self.mapping = MappingNetwork(w_dim, w_dim)
        self.generator = Generator(w_dim, num_layers, channels, img_size)
        self.discriminator = Discriminator(channels)

    def generate(self, z):
        w = self.mapping(z)
        return self.generator(w)

    def discriminate(self, image):
        return self.discriminator(image)