"""Contains PyTorch Modules"""

from torchvision.models import resnet50
import torch
from torch import nn


class EmbeddingNetwork(nn.Module):
    """Embeds lesion images into 256 dimensional embedding"""

    def __init__(self):
        super().__init__()
        self._backbone = resnet50()

        # Makes output in feature space
        self._backbone.fc = nn.Identity()

        self._embedder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: images to compute embeddings for

        Returns:
            Nx256 tensor containing embeddings, where N is the number of images.
        """
        return self._embedder(self._backbone(images))
