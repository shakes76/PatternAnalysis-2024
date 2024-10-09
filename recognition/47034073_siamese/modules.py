from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch import nn


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = resnet50()

        # Makes output in feature space
        self._backbone.fc = nn.Identity()

        self._embedder = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(num_features=1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedder(self._backbone(x))
