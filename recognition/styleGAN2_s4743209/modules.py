import torch
import torch.nn as nn
import torch.nn.functional as F


class AAConvLayer(nn.Module):
    """Attention-Augmented Convolutional Layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Self-attention components
        self.query = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.key = nn.Conv2d(out_channels, out_channels // 8, 1)
        self.value = nn.Conv2d(out_channels, out_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def attention(self, x):
        batch, C, H, W = x.size()

        # Compute query, key, value
        query = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, H * W)
        value = self.value(x).view(batch, -1, H * W)

        # Compute attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)

        return self.gamma * out + x

    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        x = self.norm(x)
        return self.activation(x)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, num_classes):
        super().__init__()
        # Class embedding
        self.class_embed = nn.Linear(num_classes, w_dim)

        # Mapping layers
        layers = []
        input_dim = z_dim + w_dim  # concatenated noise and class embedding
        for _ in range(4):  # 4 mapping layers as shown in diagram
            layers.append(nn.Linear(input_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = w_dim

        self.mapping = nn.Sequential(*layers)

    def forward(self, z, class_labels):
        # Convert class labels to one-hot
        class_onehot = F.one_hot(class_labels, num_classes=2).float()
        # Get class embedding
        class_embed = self.class_embed(class_onehot)
        # Concatenate noise and class embedding
        z_with_class = torch.cat([z, class_embed], dim=1)
        # Map to W space
        w = self.mapping(z_with_class)
        return w


class Generator(nn.Module):
    def __init__(self, style_dim, num_layers=13, channels=(512, 512, 512, 512, 256, 128, 64), img_size=256):
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
        for i, out_chan in enumerate(channels):
            self.aa_layers.append(AAConvLayer(in_chan, out_chan))
            in_chan = out_chan

        # To RGB layers
        self.to_rgb = nn.Conv2d(channels[-1], 1, 1)  # 1 channel for grayscale

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, styles, noise=None):
        batch_size = styles.size(0)

        # Start with constant input
        x = self.const.repeat(batch_size, 1, 1, 1)

        # Process through AA-Conv layers
        for i, aa_layer in enumerate(self.aa_layers):
            # Apply style modulation
            style = styles[:, i] if styles.dim() > 2 else styles
            style = style.view(batch_size, -1, 1, 1)

            x = aa_layer(x)
            x = x * (1 + style)  # Style modulation

            # Upsample after each layer except the last
            if i != len(self.aa_layers) - 1:
                x = self.upsample(x)

        # Generate final image
        x = self.to_rgb(x)

        if x.size(-1) != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512, 512, 512, 512)):
        super().__init__()
        self.channels = channels

        # Initial conv layer
        layers = [
            nn.Conv2d(1, channels[0], 3, padding=1),
            nn.LeakyReLU(0.2)
        ]

        # Downsampling conv layers
        in_chan = channels[0]
        for out_chan in channels[1:]:
            layers.extend([
                nn.Conv2d(in_chan, out_chan, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_chan),
                nn.LeakyReLU(0.2)
            ])
            in_chan = out_chan

        self.main = nn.Sequential(*layers)

        # Final classification layer
        self.final = nn.Conv2d(channels[-1], 1, 4, padding=0)

    def forward(self, x):
        x = self.main(x)
        return self.final(x).view(x.size(0), -1)


class StyleGAN2(nn.Module):
    def __init__(self, w_dim, num_layers, channels, img_size=256):
        super().__init__()
        self.mapping = MappingNetwork(w_dim, w_dim, num_classes=2)
        self.generator = Generator(w_dim, num_layers, channels, img_size)
        self.discriminator = Discriminator(channels)

    def generate(self, z, class_labels=None):
        if class_labels is None:
            # If no labels provided, randomly assign them
            class_labels = torch.randint(0, 2, (z.size(0),), device=z.device)
        # Map noise and class labels to W space
        w = self.mapping(z, class_labels)
        # Generate images
        return self.generator(w)

    def discriminate(self, image):
        return self.discriminator(image)