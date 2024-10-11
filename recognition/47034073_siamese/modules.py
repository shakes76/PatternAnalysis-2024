from torchvision.models import resnet50
import torch
from torch import nn


class EmbeddingNetwork(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedder(self._backbone(x))
