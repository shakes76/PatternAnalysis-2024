import time
import logging
from dataclasses import dataclass

from torchvision.models import efficientnet_b0
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits, cosine_similarity

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001


class TumorClassifier:
    def __init__(self, hparams: HyperParams) -> None:
        self._model = TumorTower().to(device)
        self._optim = torch.optim.Adam(
            params=self._model.parameters(), lr=hparams.learning_rate
        )
        self._hparams = hparams
        self._losses: list[float] = []
        self._benign_centroid = None
        self._malignant_centroid = None

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

    def compute_centroids(self, loader: DataLoader[tuple[torch.Tensor, int]]) -> None:
        self._benign_centroid = None
        self._malignant_centroid = None

        num_benign = 0
        num_malignant = 0
        n = 0
        start_time = time.time()
        with torch.inference_mode():
            for x, labels in loader:
                x = x.to(device)
                labels = labels.to(device)
                n += len(labels)

                embeddings = self._model.compute_embedding(x)

                if self._benign_centroid is None:
                    embedding_dim = embeddings.shape[1]
                    self._benign_centroid = torch.zeros(embedding_dim).to(device)
                if self._malignant_centroid is None:
                    self._malignant_centroid = torch.zeros(embedding_dim).to(device)

                benign = embeddings[labels == 0, :]
                malignant = embeddings[labels == 1, :]

                num_benign += len(benign)
                num_malignant += len(malignant)

                self._benign_centroid += benign.sum(dim=0)
                self._malignant_centroid += malignant.sum(dim=0)

                if time.time() - start_time > 60:
                    start_time = time.time()
                    logger.info("%d/%d", n, len(loader.dataset))

            self._benign_centroid /= num_benign
            self._malignant_centroid /= num_malignant

    def evaluate(self, loader: DataLoader) -> dict:
        with torch.inference_mode():
            hits = 0
            n = 0
            start_time = time.time()
            for x, labels in loader:
                x = x.to(device)
                labels = labels.to(device)

                n += len(labels)

                logger.debug("Shape of centroid %s", str(self._benign_centroid.shape))
                embeddings = self._model.compute_embedding(x)
                benign_distances = cosine_similarity(embeddings, self._benign_centroid)
                malignant_distances = cosine_similarity(
                    embeddings, self._malignant_centroid
                )
                logger.debug("%s", str(benign_distances.shape))
                distances = torch.stack([benign_distances, malignant_distances], dim=1)
                logger.debug("%s", str(distances.shape))
                predictions = distances.argmax(dim=1)
                logger.debug("Predictions %s", str(predictions))

                hits += torch.sum(predictions == labels)

                if time.time() - start_time > 60:
                    start_time = time.time()
                    logger.info("%d/%d", n, len(loader.dataset))

            acc = hits / n
            return acc

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
            logits = self._model(x1, x2)
            loss = binary_cross_entropy_with_logits(logits.flatten(), y)
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

        # Weighted summation of component differencess
        self._component_adder = nn.Linear(1000, 1, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        embeddings1 = self._backbone(x1)
        embeddings2 = self._backbone(x2)
        component_diffs = torch.abs(embeddings1 - embeddings2)
        distances = self._component_adder(component_diffs)
        return distances

    def compute_embedding(self, x: torch.Tensor):
        return self._backbone(x)
