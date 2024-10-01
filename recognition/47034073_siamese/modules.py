from torchvision.models import efficientnet_b0
import torch
from torch import nn
from torch.nn.functional import pairwise_distance, binary_cross_entropy_with_logits
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    num_epochs: int = 20
    batch_size: int = 32


class TumorTrainer:
    def __init__(self, hparams: HyperParams):
        self._model = TumorTower().to(device)
        self._optim = nn.optim.Adam(self._model.parameters())
        self._hparams = hparams

    def train(self, train_loader):
        for epoch in range(self._hparams.num_epochs):
            self._train_epoch(train_loader)

    def _train_epoch(self, train_loader):
        for x1, x2, y in train_loader:
            x1.to(device)
            x2.to(device)
            y.to(device)
            self._optim.zero_grad()
            distances = self._model(x1, x2)
            loss = binary_cross_entropy_with_logits(distances, y)
            loss.backward()
            self._optim.step()


class TumorTower(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = efficientnet_b0()

        # Makes output in feature space
        self._backbone.classifer = nn.Identity()

        # Weighted summation of component differences
        self._component_adder = nn.Linear(1000, 1, bias=False)

    def forward(self, x1, x2):
        embeddings1 = self._backbone(x1)
        embeddings2 = self._backbone(x2)
        component_diffs = torch.abs(embeddings1 - embeddings2)
        distances = self._component_adder(component_diffs)
        return distances
