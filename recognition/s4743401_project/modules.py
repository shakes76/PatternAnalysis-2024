import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):  # divide image into patches and apply linear projection
    def __init__(self, patch_size, in_channels, hidden_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.linear_proj = nn.Linear(patch_size * patch_size * in_channels, hidden_dim)

    def forward(self, x):
        pass

class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super(PositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, hidden_dim))

    def forward(self, x):
        pass


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        # PatchEmbedding, PositionalEmbedding, TransformerEncoder
        pass

    def forward(self, x):
        pass