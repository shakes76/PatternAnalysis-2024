from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Makes output in feature space
        self._backbone.fc = nn.Identity()

        self._embedder = nn.Sequential(
            nn.Linear(2048, 512), nn.PReLU(), nn.Linear(512, 128)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedder(self._backbone(x))
