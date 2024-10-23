"""
containing the source code of the components of GFNet model. 

Created by: Shogo Terashima
Created by:     Shogo Terashima
ID:             S47779628
Last update:    24/10/2024
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
from timm.layers import DropPath

class GlobalFilterLayer(nn.Module):
    '''
    GlobalFilterLayer is a layer that applies a learnable global filter in the frequency domain.
    Implemented based on the pseudocode of Global Filter Layer from the paper.
    It consists of Layer Normalisation, 2D FFT, Element-wise multiplication, and 2D IFFT.
    '''
    def __init__(self, height, width, dimension):
        super().__init__()
        # Define learnable global filters in the frequency domain
        w_halt = width // 2 + 1  # conjugate symmetry in FFT
        self.frequency_domain_filters = nn.Parameter(torch.randn(height, width, dimension, 2, dtype=torch.float32) * 0.02)

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
        X_tilde = X * torch.view_as_complex(self.frequency_domain_filters) # Element-wise multiplication
        x_filtered = torch.fft.irfft2(X_tilde, s=(height, width), dim=(1, 2), norm='ortho')  # inverse 2D FFT 
        x_filtered = x_filtered.reshape(batch_size, N, channels)
        return x_filtered 
class FeedForwardLayer(nn.Module):
    '''
    It consists of Layer normalisation,  two layers of MLP with a non-linear activation function, with dropout.
    '''
    def __init__(self, input_dim, drop=0.1):
        super().__init__()
        hidden_dim = input_dim * 4 # WE SET THE MLP EXPANSION RATIO TO 4 FOR ALL THE FEEDFORWARD NETWORKS (from paper)
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

class Block(nn.Module):
    '''
    Block to consist Global Filter Layer and Feed Forward Network
    '''
    def __init__(self, height, width, dimension, drop_rate=0.1, drop_path_rate=0.0, init_values=1e-5):
        super().__init__()
        self.global_filter = GlobalFilterLayer(height, width, dimension)
        self.feed_foward = FeedForwardLayer(input_dim=dimension, drop=drop_rate)
        self.drop_path = DropPath(drop_path_rate)
        self.gamma = nn.Parameter(init_values * torch.ones((dimension)), requires_grad=True)
    def forward(self, x):
        residual = x # use for skip connection
        x = self.global_filter(x) # Layer Norm -> 2D FFT -> Element-wise mult -> 2D IFFT
        x = self.feed_foward(x)  # Layer Norm -> MLP
        x = self.drop_path(self.gamma * x)        
        x = x + residual  # Skip connection
        
        return x
    

class PatchEmbedding(nn.Module):
    def __init__(self, dim_in=1, embed_dim=256, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.patch_to_embedding = nn.Conv2d(dim_in, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        
    def forward(self, x):
        x = self.patch_to_embedding(x)  # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # -> (B, embed_dim, num_patches)
        x = x.transpose(1, 2) # -> (B, num_patches, embed_dim)
        return x
    
class DownSamplingLayer(nn.Module):
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
    '''
    Here, we implemented a hierarchical version of the GFNet model.
    Each stage starts with a Patch Embedding or DownSampling Layer, followed by a sequence of Blocks.
    After the last stage, the model performs global average pooling and outputs the final classification result through a linear layer.
    '''
    def __init__(self, image_size=224, num_classes=1, blocks_per_stage=[3, 3, 10, 3], 
                 stage_dims=[64, 128, 256, 512], drop_rate=0.1, drop_path_rate=0.1, init_values=0.001, dropcls=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embedding = nn.ModuleList()
        self.position_embedding = nn.ParameterList() 
        
        patch_embedding = PatchEmbedding(dim_in=1, embed_dim=stage_dims[0], patch_size=4)
        num_patches = (image_size // 4) * (image_size // 4)
        self.patch_embedding.append(patch_embedding)
        self.position_embedding.append(nn.Parameter(torch.zeros(1, num_patches, stage_dims[0])))

        # Define DownSamplingLayers and patch embedding
        sizes = [56*image_size//224, 28*image_size//224, 14*image_size//224, 7*image_size//224] # Generally speaking, we can start from a large feature map (e.g., 56 × 56) and gradually perform downsampling after a few blocks. 

        for i in range(len(sizes)-1):
            patch_embedding = DownSamplingLayer(stage_dims[i], stage_dims[i+1])
            self.patch_embedding.append(patch_embedding)

        # storchastic drop path rate
        drop_path_probabilities = [x.item() for x in torch.linspace(0, drop_path_rate, sum(blocks_per_stage))]
        current_block_index = 0

        
        # Define the blocks for each stage
        self.blocks = nn.ModuleList()
        current_block_index = 0
        for stage_index in range(len(sizes)):
            height = sizes[stage_index]
            width = height // 2 + 1 
            num_blocks = blocks_per_stage[stage_index]
            blocks_for_current_stage = []
            for block_index in range(num_blocks):
                block = Block(
                    height=height,
                    width=width,
                    dimension=stage_dims[stage_index],
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_probabilities[current_block_index + block_index],
                    init_values=init_values
                )
                blocks_for_current_stage.append(block)
            self.blocks.append(nn.Sequential(*blocks_for_current_stage)) 
            current_block_index += num_blocks
        
        self.norm = nn.LayerNorm(stage_dims[-1])
        self.head = nn.Linear(stage_dims[-1], num_classes)  # Linear head
  
    def forward(self, x):
        # Stage 1:
        x = self.patch_embedding[0](x)
        x = x + self.position_embedding[0]
        x = self.blocks[0](x)
        
        # Stage 2:
        x = self.patch_embedding[1](x)
        x = self.blocks[1](x)

        # Stage 3:
        x = self.patch_embedding[2](x)
        x = self.blocks[2](x)

        # Stage 4:
        x = self.patch_embedding[3](x)
        x = self.blocks[3](x)
        
        x = self.norm(x)
        x = x.mean(1) # Global average pooling
        x = self.head(x)
        return x
