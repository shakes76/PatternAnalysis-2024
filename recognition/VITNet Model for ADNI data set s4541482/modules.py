# modules.py

import torch
import torch.nn as nn
import timm

class ViTModel(nn.Module):
    """
    Vision Transformer model adapted for images of size 224x224 using timm library.
    """
    def __init__(self, num_classes=2, freeze=True):
        super(ViTModel, self).__init__()
        # Initialize the ViT model with corrected image size
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=num_classes,
            img_size=224,  # Changed to 224
            in_chans=3
        )

        # Dropout for regularization
        self.model.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.embed_dim, num_classes)
        )

        # Optionally freeze layers
        if freeze:
            self.freeze_layers()

    def freeze_layers(self):
        # Freeze patch embedding
        for param in self.model.patch_embed.parameters():
            param.requires_grad = False
        # Freeze transformer blocks
        for param in self.model.blocks.parameters():
            param.requires_grad = False
        # Freeze norm layers
        for param in self.model.norm.parameters():
            param.requires_grad = False

    def unfreeze_layers(self):
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x
