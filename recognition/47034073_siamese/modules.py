from torchvision.models import efficientnet_b0
from torch import nn
from torch.nn.functional import pairwise_distance, binary_cross_entropy_with_logits
from dataclasses import dataclass


class HyperParams:
    num_epochs: int = 20


class TumorTrainer:
    def __init__(self, hparams: HyperParams):
        self._model = SiameseTumor()
        self._hparams = hparams

    def train(self, train_loader):
        for epoch in range(self._hparams.num_epochs):
            self._train_epoch(train_loader)

    def _train_epoch(self, train_loader):
        for x1, x2, y in train_loader:
            self._optimizer.zero_grad()
            distances = self._model(x1, x2)
            loss = binary_cross_entropy_with_logits(distances, targets)
            loss.backward()
            self._optimizer.step()


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
