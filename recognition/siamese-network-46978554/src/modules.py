"""Model components"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class SiameseNetwork(nn.Module):
    """A Siamese network with a ResNet backbone."""

    def __init__(self, pretrained=False):
        """
        Args:
            pretrained: Whether to use a ResNet with pretrained ImageNet weights.
        """
        super(SiameseNetwork, self).__init__()

        # Use a ResNet backbone without the last FC layer (we'll define our own)
        if pretrained:
            resnet_base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet_base = resnet50()
        self.net = nn.Sequential(
            *list(resnet_base.children())[:-1],
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=256, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=64, bias=True)
        )

        # Freeze ResNet layers if we're using pretrained weights
        if pretrained:
            for param in self.net[:-3].parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward pass through the siamese network"""
        return self.net(x)

    def forward_pair(self, x1, x2):
        """Forward pass through the siamese network for a pair of inputs"""
        return self.net(x1), self.net(x2)

