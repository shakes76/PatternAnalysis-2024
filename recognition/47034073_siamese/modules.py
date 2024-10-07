import time
import logging
from dataclasses import dataclass

from torchvision.models import resnet50
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.00001


class SiameseController:
    def __init__(self, hparams: HyperParams) -> None:
        self._model = EmbeddingNetwork().to(device)
        self._optim = torch.optim.Adam(
            params=self._model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )
        self._hparams = hparams
        self._losses: list[float] = []
        self._miner = miners.BatchHardMiner()
        self._loss = losses.TripletMarginLoss()

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
        for x, labels in train_loader:
            n += 1
            x = x.to(device)
            labels = labels.float().to(device)
            num_observations += len(labels)

            self._optim.zero_grad()
            embeddings = self._model(x)
            miner_output = self._miner(embeddings, labels)
            loss = self._loss(embeddings, labels, miner_output)

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


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = resnet50()

        # Makes output in feature space
        self._backbone.classifer = nn.Identity()

        self._embedder = nn.Sequential(nn.PReLU(), nn.Linear(1000, 128))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedder(self._backbone(x))
