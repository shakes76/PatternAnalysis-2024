import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from torch.nn import Linear, Conv2d, GroupNorm, GELU, AvgPool2d, Upsample

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_dim):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = Conv2d(out_channels, out_channels, 3, 1, padding=1)

        self.gn1 = GroupNorm(num_groups=4, num_channels=out_channels)
        self.gn2 = GroupNorm(num_groups=4, num_channels=out_channels)

        if in_channels != out_channels:
            self.shortcut = Conv2d(in_channels, out_channels, 1, 1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.fc = Linear(t_dim, 2 * out_channels)

    def forward(self, x, t):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.gn1(x)  
 
        gamma, beta = self.fc(t).chunk(2, dim=1)
        x = x * (gamma[:, :, None, None] + 1) + beta[:, :, None, None]
        x = F.silu(x)

        x = self.conv2(x)
        x = self.gn2(x) 
        x = F.silu(x)
        x = x + residual
        return x
    

def Upsample(dim, dim_out = None):
    out = dim
    if dim_out is not None:
        out = dim_out
    
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode = 'nearest'),
        Conv2d(dim, out, 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    out = dim
    if dim_out is not None:
        out = dim_out
        
    return nn.Sequential(
        AvgPool2d(2, stride=2),
        Conv2d(dim, out, 3, padding = 1)
    )

class U_net(nn.Module):
    def __init__(self, dim, dim_mults = (1, 2, 4, 8), cond_dim=10):
        super().__init__()
        self.t_dim = dim * 4
        resnet_block = partial(ResBlock, t_dim = self.t_dim + cond_dim)
        
        self.init_conv = Conv2d(1, dim, 7, padding = 3)
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        for i, (in_channel, out_channel) in enumerate(self.in_out):
            is_last = i >= (len(self.in_out) - 1)
            self.downs.append(nn.ModuleList([
                resnet_block(in_channel, in_channel),
                resnet_block(in_channel, in_channel),
                Downsample(in_channel, out_channel) if not is_last else Conv2d(in_channel, out_channel, 3, padding = 1)
            ]))
            
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_block2 = resnet_block(mid_dim, mid_dim)
        
        self.ups = nn.ModuleList([])
        for i, (in_channel, out_channel) in enumerate(reversed(self.in_out)):
            is_last = i == (len(self.in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                resnet_block(in_channel + out_channel, out_channel),
                resnet_block(in_channel + out_channel, out_channel),
                Upsample(out_channel, in_channel) if not is_last else Conv2d(out_channel, in_channel, 3, padding = 1)
            ]))
            
        
        self.out_block = resnet_block(dim, dim)
        self.outc = Conv2d(dim, 1, 1)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            Linear(dim, self.t_dim),
            GELU(),
            Linear(self.t_dim, self.t_dim)
        )
        self.cond_embed = Linear(1, cond_dim)
    
    def forward(self, x, t, c):
        x = self.init_conv(x)
        #r = x.clone()
        t = self.time_embed(t)
        c = c.unsqueeze(-1).float()
        c = self.cond_embed(c)
        t = torch.cat([t, c], dim=-1)
        
        skips = []
        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            skips.append(x)
            
            x = block2(x, t)
            skips.append(x)
            x = downsample(x)
            
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        for block1, block2, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, skips.pop()), dim = 1)
            x = block2(x, t)
            x = upsample(x)

        #x = torch.cat((x, r), dim = 1)
        x = self.out_block(x, t)
        return self.outc(x)