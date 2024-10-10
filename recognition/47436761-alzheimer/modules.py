from torchvision import datasets, transforms
import torch.nn as nn
import torch
import torch.nn.functional as F
from dataset import IMAGE_DIM, PATCH_SIZE, NUM_PATCHES, D_MODEL

def make_patch(x):
    """Divides the input images into patches."""
    # x: [batch_size, channels, height, width]
    batch_size, channels, height, width = x.size()
    patch_size = PATCH_SIZE
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    x = x.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, NUM_PATCHES, -1)
    return x

class PositionalEncoder(nn.Module):
    def __init__(self, num_patches, d_model):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model
        self.projection = nn.Linear((PATCH_SIZE**2) * 3, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, d_model, d_mlp, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout_rate)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # Self-Attention block with residual connection
        x_norm = self.layer_norm1(x)
        attn_output, _ = self.self_attention(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP block with residual connection
        x_norm = self.layer_norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        return x

class AlzheimerModel(nn.Module):
    def __init__(self, num_patches, num_layers, num_heads, d_model, d_mlp, head_layers, dropout_rate, num_classes=2):
        super(AlzheimerModel, self).__init__()

        self.pos_encoder = PositionalEncoder(num_patches, d_model)
        
        self.encoders = nn.ModuleList([
            TransformerEncoder(num_heads, d_model, d_mlp, dropout_rate) for _ in range(num_layers)
        ])
        
        # MLP Head for classification
        self.mlp = nn.Sequential(
            nn.Linear(d_model, head_layers),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(head_layers, num_classes)
        )

    def forward(self, x):
        x = make_patch(x)
        x = self.pos_encoder(x)

        # Pass through each Transformer Encoder layer
        for encoder in self.encoders:
            x = encoder(x)
        
        # Use the CLS token's output for classification
        x = x[:, 0]
        output = self.mlp(x)
        return output
