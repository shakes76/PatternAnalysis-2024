import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SinusoidalEmbeddings(nn.Module):
    """
    Sinusoidal embeddings provide positional information by using sine & cosine functions.
    """
    def __init__(self, time_steps, embed_dim):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        # sine and cosine curves of varying frequencies
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        return embeds[:, :, None, None]

class ResBlock(nn.Module):
    """
    Residual block applies convolutional layers with skip connections.
    Group norm and dropout are utilised for performance
    """
    def __init__(self, channels, groups, dropout_prob):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.gnorm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        # 3x3 same convolution
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x

class Attention(nn.Module):
    """
    Perform self-attention on the input features.
    """
    def __init__(self, channels, heads, dropout_prob):
        super().__init__()
        self.proj1 = nn.Linear(channels, channels * 3)
        self.proj2 = nn.Linear(channels, channels)
        self.num_heads = heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        # transform data from einops library to deal with tensors easily
        # (b) batch size (c) channels (h) height (w) width
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class UnetLayer(nn.Module):
    """
    A complete single layer of the UNet architecture.
    """
    def __init__(self, upscale, has_attention, num_groups, dropout_prob, num_heads, channels):
        super().__init__()
        # residual blocks used
        self.ResBlock1 = ResBlock(channels=channels, groups=num_groups, dropout_prob=dropout_prob)
        self.ResBlock2 = ResBlock(channels=channels, groups=num_groups, dropout_prob=dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        if has_attention:
            self.attention_layer = Attention(channels, heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x

class UNET(nn.Module):
    """
    U-Net model for image segmentation and denoising tasks.
    32 groups are used for group normalization.
    8 heads are used for self-attention.
    dropout probability is set to 0.1.
    """
    def __init__(self,
                 channels = [128, 256, 512, 1024, 1024, 768],
                 is_attention = [False, True, False, False, False, True],
                 is_upscale = [False, False, False, True, True, True],
                 num_groups = 32,
                 dropout_prob = 0.1,
                 num_heads = 8,
                 input_channels = 3,
                 output_channels = 3,
                 time_steps = 1000):
        super().__init__()
        self.num_layers = len(channels)
        self.shallow_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)
        out_channels = (channels[-1] // 2) + channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.embeddings = SinusoidalEmbeddings(time_steps=time_steps, embed_dim=max(channels))
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=is_upscale[i],
                has_attention=is_attention[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                channels=channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        """
        forward step
        """
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            x, r = layer(x, embeddings)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))

class DiffusionScheduler(nn.Module):
    """
    Linearly increase the beta values from 1e-4 to 0.02 over the entire training process.
    """
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]