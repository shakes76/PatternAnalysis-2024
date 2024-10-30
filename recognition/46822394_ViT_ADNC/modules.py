"""
Author: Ella WANG

Contains the source code for the components of ViT classifying the Alzheimer’s disease (normal and AD) of the ADNI brain data
Each component is implementated as a class or a function.
"""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16
import torch.nn.functional as F

class ViTClassifier(nn.Module):
    """
    Modified Vision Transformer for grayscale images
    """
    def __init__(self, num_classes=4):
        super(ViTClassifier, self).__init__()

        # Load the pre-trained ViT model
        self.vit = vit_b_16(pretrained=True)

        # Modify the first layer to accept grayscale input
        # Create new patch embedding layer with 1 input channel instead of 3
        new_patch_embed = nn.Conv2d(
            in_channels=1,  # Changed from 3 to 1 for grayscale
            out_channels=768,
            kernel_size=16,
            stride=16
        )

        # Initialize the weights of the new layer
        # Average the weights across the RGB channels
        with torch.no_grad():
            new_patch_embed.weight = nn.Parameter(
                self.vit.conv_proj.weight.sum(dim=1, keepdim=True) / 3.0
            )
            new_patch_embed.bias = nn.Parameter(self.vit.conv_proj.bias)

        # Replace the patch embedding layer
        self.vit.conv_proj = new_patch_embed

        # Modify the classifier head for our number of classes
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vit(x)

class EnhancedViTClassifier(nn.Module):
    """
    Enhanced Vision Transformer for grayscale medical images with:
    - Data normalization
    - Dropout for regularization
    - Feature augmentation
    - Residual connections
    - Label smoothing support
    - Proper image size handling
    """
    def __init__(self, num_classes=4, dropout_rate=0.2, feature_dropout=0.1, image_size=224):
        super(EnhancedViTClassifier, self).__init__()
        self.image_size = image_size

        # Load the pre-trained ViT model
        self.vit = vit_b_16(pretrained=True)

        # Add positional dropout separately
        self.pos_dropout = nn.Dropout(0.1)

        # Create our own class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        nn.init.normal_(self.class_token, std=0.02)  # Initialize following ViT paper

        # Calculate number of patches
        self.patch_size = 16
        self.num_patches = (image_size // self.patch_size) ** 2

        # Modify position embedding for our sequence length
        old_pos_embed = self.vit.encoder.pos_embedding
        new_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, old_pos_embed.shape[2])
        )

        # Initialize new position embeddings
        # Copy the class token position embedding
        new_pos_embed.data[:, 0] = old_pos_embed.data[:, 0]

        # Resize patch position embeddings
        pos_tokens = old_pos_embed.data[:, 1:]
        pos_tokens = pos_tokens.reshape(-1, 14, 14, old_pos_embed.shape[2])
        pos_tokens = F.interpolate(
            pos_tokens.permute(0, 3, 1, 2),
            size=(image_size // self.patch_size, image_size // self.patch_size),
            mode='bicubic',
            align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed.data[:, 1:] = pos_tokens

        self.vit.encoder.pos_embedding = new_pos_embed

        # Modify first layer for grayscale input
        new_patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=768,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Initialize weights with scaled averaging of RGB weights
        with torch.no_grad():
            rgb_weights = self.vit.conv_proj.weight
            grayscale_weights = (0.2989 * rgb_weights[:, 0:1, :, :] +
                               0.5870 * rgb_weights[:, 1:2, :, :] +
                               0.1140 * rgb_weights[:, 2:3, :, :])
            new_patch_embed.weight = nn.Parameter(grayscale_weights)
            new_patch_embed.bias = nn.Parameter(self.vit.conv_proj.bias)

        self.vit.conv_proj = new_patch_embed

        # Feature extraction layers
        num_features = self.vit.heads.head.in_features
        self.feature_dropout = nn.Dropout(feature_dropout)

        # Additional layers for better feature representation
        self.feature_enhancement = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Final classification layers with dropout
        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.LayerNorm(num_features // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features // 2, num_classes)
        )

        # Initialize new layers
        self._initialize_weights()

        # Batch normalization for input
        self.input_norm = nn.BatchNorm2d(1)

        # Image preprocessing
        self.register_buffer(
            'mean', torch.tensor([0.5]).view(1, 1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.5]).view(1, 1, 1, 1)
        )

    def _initialize_weights(self):
        """Initialize the weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def preprocess(self, x):
        """Preprocess input images"""
        # Ensure correct size
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                            mode='bilinear', align_corners=False)

        # Normalize
        x = (x - self.mean) / self.std
        return x

    def forward(self, x, label_smoothing=0.1):
        # Preprocess input
        x = self.preprocess(x)

        # Input normalization
        x = self.input_norm(x)

        # Get ViT features
        x = self.vit.conv_proj(x)

        # Reshape to sequence
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.num_patches, -1)

        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        # Add position embeddings and apply dropout
        x = x + self.vit.encoder.pos_embedding
        x = self.pos_dropout(x)

        # Pass through encoder
        x = self.vit.encoder.layers(x)

        # Get class token output
        features = x[:, 0]

        # Feature enhancement with residual connection
        enhanced_features = self.feature_enhancement(features)
        features = features + enhanced_features

        # Apply feature dropout
        features = self.feature_dropout(features)

        # Final classification
        logits = self.classifier(features)
        return logits

    def get_attention_weights(self):
        """Extract attention weights for visualization"""
        return self.vit.encoder.layers[-1].self_attention.attention_probs