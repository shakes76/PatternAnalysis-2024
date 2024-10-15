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
    def __init__(self, drop_prob=0.0, is_training=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.is_training = is_training

    def forward(self, x):
        if self.drop_prob == 0. or not self.is_training:
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
    def __init__(self, height, width, dimension, mlp_drop=0.1, drop_path_rate=0.0, is_training=False):
        super().__init__()
        self.global_filter = GlobalFilterLayer(height, width, dimension)
        self.mlp = MLP(dimension, drop=mlp_drop)
        self.drop_path = DropPath(drop_prob=drop_path_rate, is_training = is_training) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        residual = x # use for skip connection
        x = self.global_filter(x) # Layer Norm -> 2D FFT -> Element-wise mult -> 2D IFFT
        x = self.mlp(x)  # Layer Norm -> MLP
        x = self.drop_path(x)        
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
    def __init__(self, img_size=224, num_classes=2, embed_dim=64, num_blocks=[3, 3, 10, 3], dims=[64, 128, 256, 512], drop_rate=0.1, drop_path_rate=0.1, is_training = False):
        super().__init__()

        self.patch_embed = PatchEmbedding(dim_in=1, embed_dim=embed_dim, patch_size=4)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // 4) * (img_size // 4), embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Define the stages
        self.stage1 = nn.ModuleList([Block(img_size // 4, img_size // 4, dims[0], mlp_drop=drop_rate, drop_path_rate=drop_path_rate, is_training=is_training) for i in range(num_blocks[0])])
        self.down1 = Downlayer(dims[0], dims[1])

        self.stage2 = nn.ModuleList([Block(img_size // 8, img_size // 8, dims[1], mlp_drop=drop_rate, drop_path_rate=drop_path_rate, is_training=is_training) for i in range(num_blocks[1])])
        self.down2 = Downlayer(dims[1], dims[2])

        self.stage3 = nn.ModuleList([Block(img_size // 16, img_size // 16, dims[2], mlp_drop=drop_rate, drop_path_rate=drop_path_rate, is_training=is_training) for i in range(num_blocks[2])])
        self.down3 = Downlayer(dims[2], dims[3])

        self.stage4 = nn.ModuleList([Block(img_size // 32, img_size // 32, dims[3], mlp_drop=drop_rate, drop_path_rate=drop_path_rate, is_training=is_training) for i in range(num_blocks[3])])

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)  # Linear head

    def forward(self, x):
        # Stage 1:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.stage1:
            x = block(x)
        x = self.down1(x)

        # Stage 2
        for block in self.stage2:
            x = block(x)
        x = self.down2(x)

        # Stage 3
        for block in self.stage3:
            x = block(x)
        x = self.down3(x)

        # Stage 4
        for block in self.stage4:
            x = block(x)

        # Global average pooling
        x = x.mean(1)  # (B, num_patches, embed_dim)
        
        #x = self.norm(x)
        x = self.head(x)
        return x


# tiny_model = GFNet(
#     img_size=224,
#     num_classes=2,
#     embed_dim=64,
#     num_blocks=[3, 3, 10, 3],
#     dims=[64, 128, 256, 512],
#     drop_rate=0.1,
#     drop_path_rate=0.1,
#     is_training=True
# )

# test_model = GFNet(
#     img_size=224,
#     num_classes=2,
#     embed_dim=64,
#     num_blocks=[1, 1, 1, 1],
#     dims=[32, 64, 128, 256],
#     drop_rate=0.05,
#     drop_path_rate=0.05,
#     is_training=True
# )