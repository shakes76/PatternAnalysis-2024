import time
import logging
from dataclasses import dataclass

from torchvision.models import efficientnet_b0
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import (
    binary_cross_entropy_with_logits,
    cosine_similarity,
    cosine_embedding_loss,
)

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.00001


class TumorClassifier:
    def __init__(self, hparams: HyperParams) -> None:
        self._model = TumorTower().to(device)
        self._optim = torch.optim.Adam(
            params=self._model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )
        self._hparams = hparams
        self._losses: list[float] = []

    def train(
        self, train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]]
    ) -> None:
        for epoch in range(self._hparams.num_epochs):
            self._train_epoch(train_loader)
            logger.info("Epoch %d / loss %e", epoch, self._losses[-1])

    def compute_all_embeddings(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            all_embeddings = []
            all_labels = []
            for x, labels in loader:
                x = x.to(device)
                embeddings = self._model.compute_embedding(x).cpu()
                all_embeddings.append(embeddings)
                all_labels.append(labels)

            return torch.cat(all_embeddings), torch.cat(all_labels)

    def _train_epoch(
        self, train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]]
    ) -> None:
        start_time = time.time()
        avg_loss = 0.0
        n = 0
        num_observations = 0
        for x1, x2, y in train_loader:
            n += 1
            logger.debug("start train loop")
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.float().to(device)
            num_observations += len(y)

            self._optim.zero_grad()
            # logits = self._model(x1, x2)
            # loss = binary_cross_entropy_with_logits(logits.flatten(), y)
            embed1 = self._model.compute_embedding(x1)
            embed2 = self._model.compute_embedding(x2)

            # Turn 0 targets into -1 for cosine_embedding_loss
            y = y - (y == 0).float()

            loss = cosine_embedding_loss(embed1, embed2, y)
            avg_loss += loss.item()
            loss.backward()
            self._optim.step()

            if time.time() - start_time > 60:
                start_time = time.time()
                logger.info(
                    "step %d / loss %e / progress %d/%d",
                    n,
                    avg_loss / n,
                    num_observations,
                    len(train_loader.dataset),
                )

        avg_loss /= n
        self._losses.append(avg_loss)


class TumorTower(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = efficientnet_b0()

        # Makes output in feature space
        self._backbone.classifer = nn.Identity()

        self._embedder = nn.Sequential(nn.PReLU(), nn.Linear(1000, 128))

        # Weighted summation of component differencess
        self._component_adder = nn.Linear(128, 1, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        embeddings1 = self._embedder(self._backbone(x1))
        embeddings2 = self._embedder(self._backbone(x2))
        component_diffs = torch.abs(embeddings1 - embeddings2)
        distances = self._component_adder(component_diffs)
        return distances

    def compute_embedding(self, x: torch.Tensor):
        return self._embedder(self._backbone(x))
