import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.n_patches = (256 // patch_size) * (240 // patch_size)

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)  # Flatten
        x = x.transpose(1, 2)  # Adjust dimensions to fit the Transformer encoder
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, mlp_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerNet(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_channels=1,
        num_classes=2,
        embed_dim=768,
        num_heads=16,
        depth=8,
        mlp_dim=2048,
        dropout=0.1,
    ):
        super(TransformerNet, self).__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Learnable class token
        self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, (256 // patch_size) * (240 // patch_size) + 1, embed_dim)
        )

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            embed_dim, num_heads, depth, mlp_dim, dropout
        )

        # Global Convolutional Layer from GFNet
        self.global_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)

        # Class token
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)

        # Concatenate class token with patch embeddings
        x = torch.cat((class_token, x), dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Extract the class token
        class_token_final = x[:, 0]

        # Global Convolution Layer (GFNet)
        class_token_final = class_token_final.unsqueeze(2).unsqueeze(3)  # Reshape to 4D
        class_token_final = self.global_conv(class_token_final)  # Global convolution
        class_token_final = class_token_final.squeeze()  # Flatten back to 2D

        # Classification head
        output = self.fc(class_token_final)

        return output
