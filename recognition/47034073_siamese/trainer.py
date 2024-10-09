import time
import logging
import pathlib
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners, distances, regularizers

from modules import EmbeddingNetwork

MODEL_DIR = pathlib.Path("models")

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.000001
    margin: float = 0.2


class SiameseController:
    def __init__(self, hparams: HyperParams, model_name: str) -> None:
        self._model = EmbeddingNetwork().to(device)
        self._optim = torch.optim.Adam(
            params=self._model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )
        self._hparams = hparams
        self._losses: list[float] = []
        self._distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        self._miner = miners.TripletMarginMiner(
            margin=hparams.margin, type_of_triplets="semihard", distance=self._distance
        )
        self._loss = losses.TripletMarginLoss(
            margin=hparams.margin,
            distance=self._distance,
            embedding_regularizer=regularizers.LpRegularizer(),
        )
        self._epoch = 0
        self._model_name = model_name

    def train(self, train_loader: DataLoader) -> None:
        # logger.info("Using semihard triplets")
        for _ in range(self._hparams.num_epochs):
            self._train_epoch(train_loader)
            logger.info("Epoch %d / loss %e", self._epoch, self._losses[-1])

            self._epoch += 1
            self.save_model(self._model_name)

    def compute_all_embeddings(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            all_embeddings = []
            all_labels = []
            for x, labels in loader:
                x = x.to(device)
                embeddings = self._model(x).cpu().detach()
                all_embeddings.append(embeddings)
                all_labels.append(labels.detach())

            return torch.cat(all_embeddings), torch.cat(all_labels)

    def save_model(self, name: str) -> None:
        MODEL_DIR.mkdir(exist_ok=True)

        state = {
            "model_state": self._model.state_dict(),
            "optim_state": self._optim.state_dict(),
            "losses": self._losses,
            "epoch": self._epoch,
            "hparams": self._hparams,
        }

        torch.save(state, MODEL_DIR / f"{name}.pt")

    def load_model(self, name: str) -> None:
        state = torch.load(MODEL_DIR / f"{name}.pt", weights_only=False)

        self._model.load_state_dict(state["model_state"])
        self._optim.load_state_dict(state["optim_state"])
        self._losses = state["losses"]
        self._epoch = state["epoch"]

    def _train_epoch(
        self, train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]]
    ) -> None:
        start_time = time.time()
        avg_loss = 0.0
        n = 0
        num_observations = 0
        for x, labels in train_loader:
            self._optim.zero_grad()

            if n == 0:
                logger.debug("labels %s\n%s", labels.shape, labels)
            n += 1
            x = x.to(device)
            labels = labels.to(device)
            num_observations += len(labels)

            embeddings = self._model(x)
            hard_triplets = self._miner(embeddings, labels)
            loss = self._loss(embeddings, labels, hard_triplets)

            avg_loss += loss.item()
            loss.backward()
            self._optim.step()

            if time.time() - start_time > 60:
                start_time = time.time()
                logger.info(
                    "step %d / loss %e / progress %d / num mined triplets %d",
                    n,
                    avg_loss / n,
                    num_observations,
                    self._miner.num_triplets if self._miner is not None else -1,
                )

        avg_loss /= n
        self._losses.append(avg_loss)
