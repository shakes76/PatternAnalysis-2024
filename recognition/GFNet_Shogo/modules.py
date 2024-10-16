"""
containing the source code of the components of your model. Each component must be implementated as a class or a function

Created by: Shogo Terashima

Reference:
@inproceedings{rao2021global,
  title={Global Filter Networks for Image Classification},
  author={Rao, Yongming and Zhao, Wenliang and Zhu, Zheng and Lu, Jiwen and Zhou, Jie},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}
"""
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
class GlobalFilterLayer(nn.Module):
    '''
    Implemeted based on the pseudocode of Global Filter Layer from the paper.
    '''
    def __init__(self, height, width, dimension):
        super().__init__()
        # Define learnable global filters in the frequency domain
        w_halt = width // 2 + 1  # conjugate symmetry in FFT
        self.frequency_domain_filters = nn.Parameter(
            torch.randn(height, w_halt, dimension, dtype=torch.cfloat) * 0.02
        ) # Fourier transform output is complex

        self.layer_norm = nn.LayerNorm(dimension)

    def forward(self, x):
        '''
        layer norm -> 2D FFT -> element wise mult -> 2D IFFT 
        '''
        batch_size, N, channels = x.shape
        height = width = int(N ** 0.5)  # assuming N is a perfect square
        
        x = x.view(batch_size, height, width, channels)
        x_normalized = self.layer_norm(x)
        X = torch.fft.rfft2(x_normalized, dim=(1, 2), norm='ortho') # 2D FFT 
        X_tilde = X * self.frequency_domain_filters # Element-wise multiplication
        x_filtered = torch.fft.irfft2(X_tilde, s=(height, width), dim=(1, 2), norm='ortho')  # inverse 2D FFT 
        x_filtered = x_filtered.reshape(batch_size, N, channels)
        return x_filtered 
class MLP(nn.Module):
    '''
    two layers with a non-linear activation function in between
    '''
    def __init__(self, input_dim, hidden_dim=None, drop=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 4 # WE SET THE MLP EXPANSION RATIO TO 4 FOR ALL THE FEEDFORWARD NETWORKS (from paper)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(input_dim) 

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class DropPath(nn.Module):
    '''
    Reference: https://stackoverflow.com/questions/69175642/droppath-in-timm-seems-like-a-dropout
    '''
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x / keep_prob * random_tensor
        return output
    
class Block(nn.Module):
    '''
    Block to consist Global Filter Layer and Feed Forward Network
    '''
    def __init__(self, height, width, dimension, mlp_drop=0.1, drop_path_rate=0.0, init_values=1e-5):
        super().__init__()
        self.global_filter = GlobalFilterLayer(height, width, dimension)
        self.mlp = MLP(input_dim=dimension, drop=mlp_drop)
        self.drop_path = DropPath(drop_prob=drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(init_values * torch.ones((dimension)), requires_grad=True)
    def forward(self, x):
        residual = x # use for skip connection
        x = self.global_filter(x) # Layer Norm -> 2D FFT -> Element-wise mult -> 2D IFFT
        x = self.mlp(x)  # Layer Norm -> MLP
        x = self.drop_path(self.gamma * x)        
        x = x + residual  # Skip connection
        
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, dim_in=1, embed_dim=256, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # -> (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # -> (B, num_patches, embed_dim)
        return x
    
class Downlayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsample = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)

    def forward(self, x):
        batch_size, N, channels = x.size()
        height = width = int(N ** 0.5)  # assume 224 x 224
        x = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2) # to (B, C, H, W)
        x = self.downsample(x).permute(0, 2, 3, 1) # to (B, dim_out, H/2, W/2)
        x = x.reshape(batch_size, -1, self.dim_out)  # (B, num_patches, dim_out)
        return x
class GFNet(nn.Module):
    def __init__(self, img_size=224, num_classes=1, initial_embed_dim=64, blocks_per_stage=[3, 3, 10, 3], 
                 stage_dims=[64, 128, 256, 512], drop_rate=0.1, drop_path_rate=0.1, init_values=0.001, dropcls=0.0):
        super().__init__()

        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ParameterList() 
        
        patch_embed = PatchEmbedding(dim_in=1, embed_dim=initial_embed_dim, patch_size=4)
        num_patches = (img_size // 4) * (img_size // 4)
        self.patch_embed.append(patch_embed)
        self.pos_embed.append(nn.Parameter(torch.zeros(1, num_patches, initial_embed_dim)))

        # Define DownLayers and patch embedding
        sizes = [56, 28, 14, 7]
        for i in range(1, len(sizes)):
            downlayer = Downlayer(stage_dims[i-1], stage_dims[i])
            self.patch_embed.append(downlayer)
            num_patches = (sizes[i] * sizes[i])
            self.pos_embed.append(nn.Parameter(torch.zeros(1, num_patches, stage_dims[i])))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Define the blocks for each stage
        self.blocks = nn.ModuleList([
            nn.ModuleList([Block(sizes[i], sizes[i], stage_dims[i], mlp_drop=drop_rate, drop_path_rate=drop_path_rate, 
                                 init_values=init_values) for _ in range(blocks_per_stage[i])])
            for i in range(4)
        ])

        self.norm = nn.LayerNorm(stage_dims[-1])
        self.head = nn.Linear(stage_dims[-1], num_classes)  # Linear head

        # Head dropout
        self.final_dropout = nn.Dropout(p=dropcls) if dropcls > 0 else nn.Identity()

    def forward(self, x):
        # Stage 1:
        x = self.patch_embed[0](x)
        x = x + self.pos_embed[0]
        x = self.pos_drop(x)

        for block in self.blocks[0]:
            x = block(x)
        x = self.patch_embed[1](x)  # Downsample and patch embed for stage 2

        # Stage 2:
        x = x + self.pos_embed[1]
        for block in self.blocks[1]:
            x = block(x)
        x = self.patch_embed[2](x)  # Downsample and patch embed for stage 3

        # Stage 3:
        x = x + self.pos_embed[2]
        for block in self.blocks[2]:
            x = block(x)
        x = self.patch_embed[3](x)  # Downsample and patch embed for stage 4

        # Stage 4:
        x = x + self.pos_embed[3]
        for block in self.blocks[3]:
            x = block(x)

        # Global average pooling
        x = x.mean(1)  # (B, num_patches, embed_dim)

        # normalization -> head dropout -> classification
        x = self.norm(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x
