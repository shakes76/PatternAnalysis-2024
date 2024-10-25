# Contains the source code of the Vision Transformer

import torch
import torch.nn as nn

# Define the TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention layer
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        
        # Layer normalization for the input and output of the attention layer
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MLP block with two linear layers and GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),  # First linear layer
            nn.GELU(),                 # Activation function
            nn.Dropout(dropout),       # Dropout for regularization
            nn.Linear(mlp_dim, mlp_dim),  # Added layer for increased capacity
            nn.GELU(),                 # Second activation
            nn.Dropout(dropout),       # Dropout for regularization
            nn.Linear(mlp_dim, dim)    # Final layer to project back to original dimension
        )

    def forward(self, x):
        # Compute self-attention output
        attn_output, _ = self.attn(x, x, x)
        
        # Add attention output to the input (residual connection) and apply normalization
        x = self.norm1(x + attn_output)
        
        # Add MLP output to the input (residual connection) and apply normalization
        x = self.norm2(x + self.mlp(x))
        
        return x

# Define the Vision Transformer Model
class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=32, num_classes=2, dim=128, depth=8, heads=8, mlp_dim=256):
        super(SimpleViT, self).__init__()
        
        # Calculate the number of patches and the dimension of each patch
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size ** 2  # Assuming RGB images

        # Linear layer for embedding patches into a higher dimensional space
        self.patch_embedding = nn.Linear(self.patch_dim, dim)
        
        # Positional embedding to retain spatial information
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Class token for classification, which will be used in the final layer
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Stack of transformer blocks
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)
        ])

        # Identity layer to use the class token as input to the final classifier
        self.to_cls_token = nn.Identity()
        
        # MLP head for final classification
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),         # Normalization before the final classification
            nn.Linear(dim, num_classes)  # Final layer to output class probabilities
        )

    def forward(self, x):
        # Get the batch size
        batch_size = x.size(0)
        
        # Extract patches from the input image tensor
        patches = x.unfold(2, 32, 32).unfold(3, 32, 32)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, self.num_patches, -1)
        
        # Embed the patches
        x = self.patch_embedding(patches)

        # Expand class token to the batch size and concatenate with embedded patches
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) + self.pos_embedding  # Add positional embedding

        # Pass through each transformer block
        for layer in self.transformer:
            x = layer(x)

        # Select the class token for final classification
        x = self.to_cls_token(x[:, 0])
        
        return self.mlp_head(x)  # Output class probabilities

# Function to get the Vision Transformer model
def get_vit_model(num_classes=2):
    return SimpleViT(num_classes=num_classes)



