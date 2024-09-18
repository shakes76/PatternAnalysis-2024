import torch
import torch.nn as nn
import torch.nn.functional as F

"""
<<<<< Diffusion Model >>>>>
"""
class StableDiffusion(nn.Module):
    """
    Advanced Stable Diffusion model. Customisable encoder and decoder properties.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        model_channels (int): Number of channels in the model
        num_res_blocks (int): Number of residual blocks
        attention_resolutions (list): List of attention resolutions
        channel_mult (list): List of channel multipliers
        num_heads (int): Number of attention heads    
    """
    def __init__(self,
                in_channels=3,
                out_channels=3,
                model_channels=256,
                num_res_blocks=2,
                attention_resolutions=(16,8),
                channel_mult=(1, 2, 4, 8),
                num_heads=8):
        super(StableDiffusion, self).__init__()

        self.encoder = Encoder(
            in_channels=in_channels, 
            model_channels=model_channels, 
            num_res_blocks=num_res_blocks, 
            attention_resolutions=attention_resolutions, 
            channel_mult= channel_mult, 
            num_heads=num_heads
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult[::-1],
            num_heads=num_heads
        )

        # Time embedding - allows for longer sequences
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )


    def forward(self, x):

        # Time embedding
        t_embed = self.time_embed(x)

        # Encoder
        h = self.encoder(x)

        # Add time embedding
        h += t_embed[:, :, None, None]

        # Decoder
        out = self.decoder(h)

        return out
    

"""
<<<<< Encoder and Decoder >>>>>
"""
class Encoder(nn.Module):
    """
    Advanced encoder for the Stable Diffusion model.

    - Residual blocks for improved gradient flow in deep networks.
    - Self-attention mechanisms to capture long-range dependencies in images.
    - Progressive downsampling to efficiently capture multi-scale features.
    - Group Normalization for consistent performance across various batch sizes.
    - SiLU (Swish) activation for enhanced non-linearity and gradient propagation.

    Args:

        in_channels (int): Number of input channels
        model_channels (int): Number of channels in the model
        num_res_blocks (int): Number of residual blocks
        attention_resolutions (list): List of attention resolutions
        channel_mult (list): List of channel multipliers
        num_heads (int): Number of attention heads  
    """
    def __init__(self,
                 in_channels=3,
                 model_channels=256,
                 num_res_blocks=2,
                 attention_resolutions=(16,8),
                 channel_mult=(1, 2, 4, 8),
                 num_heads=8):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, stride=1, padding=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([])
        channels = [model_channels * mult for mult in channel_mult]

        in_chan = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                out_chan = channels[level]
                self.encoder_blocks.append(ResidualBlock(in_chan, out_chan))
                in_chan = out_chan
                if mult in attention_resolutions:
                    self.encoder_blocks.append(AttentionBlock(in_chan))
            
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(DownsampleBlock(in_chan, in_chan))

        # Normalisation and convolution
        self.norm_out = nn.GroupNorm(32, in_chan)
        self.conv_out = nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x
    

class Decoder(nn.Module):
    """
    Advanced decoderr for the Stable Diffusion model.

    - Residual blocks for improved gradient flow in deep networks.
    - Self-attention mechanisms to capture long-range dependencies in images.
    - Progressive upsampling to efficiently represent multi-scale features.
    - Group Normalization for consistent performance across various batch sizes.
    - SiLU (Swish) activation for enhanced non-linearity and gradient propagation.

    Args:
        out_channels (int): Number of output channels
        model_channels (int): Number of channels in the model
        num_res_blocks (int): Number of residual blocks
        attention_resolutions (list): List of attention resolutions
        channel_mult (list): List of channel multipliers
        num_heads (int): Number of attention heads
    """
    def __init__(self, out_channels=3, model_channels=256, num_res_blocks=2, attention_resolutions=(16,8), channel_mult=(8, 4, 2, 1), num_heads=8):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        # Initial convolution
        self.conv_in = nn.Conv2d(model_channels * channel_mult[0], model_channels * channel_mult[0], kernel_size=3, stride=1, padding=1)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([])
        channels = [model_channels * mult for mult in channel_mult]

        in_chan = model_channels * channel_mult[0]
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks + 1):
                out_chan = channels[level]
                self.decoder_blocks.append(ResidualBlock(in_chan, out_chan))
                in_chan = out_chan
                if mult in attention_resolutions:
                    self.decoder_blocks.append(AttentionBlock(in_chan))
            
            if level != len(channel_mult) - 1:
                self.decoder_blocks.append(UpsampleBlock(in_chan, channels[level + 1]))
                in_chan = channels[level + 1]

        # Normalisation and convolution
        self.norm_out = nn.GroupNorm(32, in_chan)
        self.conv_out = nn.Conv2d(in_chan, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)

        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


"""
<<<<< Blocks >>>>>
"""
class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and SiLU activation

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the convolutional layers
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
        )
        self.silu = nn.SiLU(inplace=True)
        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(32, out_channels),
            )
    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.silu(out)
        return out
    

class AttentionBlock(nn.Module):
    """
    AttentionBlock block for the decoder

    Args:
        channels (int): Number channels for attention block
    """
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=8, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x 
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)
        

class DownsampleBlock(nn.Module):
    """
    Downsample block for the encoder

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.GroupNorm(32, out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        return x
    

class UpsampleBlock(nn.Module):
    """
    Upsample for the diffusion model

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.GroupNorm(32, out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.silu(x)
        return x
