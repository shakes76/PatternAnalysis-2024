from torchvision.models import efficientnet_b0
from torch import nn
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
            embeddings1 = self._model(x1)
            embeddings2 = self._model(x2)


class SiameseTumor(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = efficientnet_b0()

        # Makes output in feature space
        self._backbone.classifer = nn.Identity()

    def forward(self, x):
        return self._backbone(x)
