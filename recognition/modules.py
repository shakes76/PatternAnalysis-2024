"""
Contains the source code for the components of GFNet classifying the Alzheimer’s disease (normal and AD) of the ADNI brain data
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
    """
    def __init__(self, num_classes=4, dropout_rate=0.2, feature_dropout=0.1):
        super(EnhancedViTClassifier, self).__init__()

        # Load the pre-trained ViT model
        self.vit = vit_b_16(pretrained=True)

        # Modify first layer for grayscale input with careful initialization
        new_patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=768,
            kernel_size=16,
            stride=16
        )

        # Initialize weights with scaled averaging
        with torch.no_grad():
            rgb_weights = self.vit.conv_proj.weight
            # Use sophisticated weight initialization for grayscale
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

    def _initialize_weights(self):
        """Initialize the weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, label_smoothing=0.1):
        # Input normalization
        x = self.input_norm(x)

        # Get ViT features
        features = self.vit.encoder(self.vit._process_input(x))
        features = self.vit.heads.head_drop(features[:, 0])

        # Feature enhancement with residual connection
        enhanced_features = self.feature_enhancement(features)
        features = features + enhanced_features  # Residual connection

        # Apply feature dropout
        features = self.feature_dropout(features)

        # Final classification
        logits = self.classifier(features)

        return logits

    def get_attention_weights(self):
        """Extract attention weights for visualization"""
        return self.vit.encoder.layers[-1].self_attention.attention_probs