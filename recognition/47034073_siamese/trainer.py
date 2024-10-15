"""Training loop, saving and loading the model."""

import time
import logging
import pathlib
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners, distances

from modules import EmbeddingNetwork

MODEL_DIR = pathlib.Path("models")

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    """Training hyperparameters"""

    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.000001
    margin: float = 0.2


class SiameseController:
    """Facilitates training, saving and loading."""

    def __init__(self, hparams: HyperParams, model_name: str) -> None:
        """
        Args:
            hparams: Hyperparameters.
            model_name: Used for saving the model
        """
        self._model = EmbeddingNetwork().to(device)
        self._optim = torch.optim.Adam(
            params=self._model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )
        self._hparams = hparams
        self.losses: list[float] = []
        self.mined_each_step: list[int] = []
        self._epoch = 0
        self._model_name = model_name

        self._distance = distances.LpDistance(normalize_embeddings=True, p=2, power=1)
        self._miner = miners.TripletMarginMiner(
            margin=hparams.margin, type_of_triplets="all", distance=self._distance
        )
        self._loss = losses.TripletMarginLoss(
            margin=hparams.margin,
            distance=self._distance,
        )

        self.end_of_epoch_func = lambda: None

    def train(self, train_loader: DataLoader) -> None:
        """Train the model.
        Args:
            train_loader: DataLoader which iterates over training data.
        """
        for _ in range(self._hparams.num_epochs):
            self._train_epoch(train_loader)
            logger.info("Epoch %d / loss %e", self._epoch, self.losses[-1])

            self._epoch += 1
            self.save_model(self._model_name)

            self.end_of_epoch_func()

    def compute_all_embeddings(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            DataLoader: Should iterate over images for which to compute all embeddings for.

        Returns:
            Let N be the number of observations and d be the dimension of embeddings.
                The first element in tuple is Nxd tensor containing embeddings.
                Second element is length N tensor containing all labels.
        """
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
        """
        Args:
            name: Model will be saved to models/{name}.pt
        """
        MODEL_DIR.mkdir(exist_ok=True)

        state = {
            "model_state": self._model.state_dict(),
            "optim_state": self._optim.state_dict(),
            "losses": self.losses,
            "epoch": self._epoch,
            "hparams": self._hparams,
            "mined_each_step": self.mined_each_step,
        }

        torch.save(state, MODEL_DIR / f"{name}.pt")

    def load_model(self, name: str) -> None:
        """
        Args:
            name: Model will be loaded from models/{name}.pt
        """
        state = torch.load(MODEL_DIR / f"{name}.pt", weights_only=False)

        self._model.load_state_dict(state["model_state"])
        self._optim.load_state_dict(state["optim_state"])
        self.losses = state["losses"]
        self.mined_each_step = state.get("mined_each_step")
        if self.mined_each_step is None:
            self.mined_each_step = []
        self._epoch = state["epoch"]

    def _train_epoch(
        self, train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, int]]
    ) -> None:
        start_time = time.time()
        avg_loss = 0.0
        num_steps = 0
        num_observations = 0
        for x, labels in train_loader:
            self._optim.zero_grad()

            num_steps += 1
            x = x.to(device)
            labels = labels.to(device)
            num_observations += len(labels)

            embeddings = self._model(x)
            hard_triplets = self._miner(embeddings, labels)
            self.mined_each_step.append(self._miner.num_triplets)
            loss = self._loss(embeddings, labels, hard_triplets)

            avg_loss += loss.item()
            loss.backward()
            self._optim.step()

            if time.time() - start_time > 60:
                start_time = time.time()
                logger.info(
                    "epoch %d / step %d / loss %e / progress %d / num mined %d",
                    self._epoch,
                    num_steps,
                    loss.item(),
                    num_observations,
                    self._miner.num_triplets,
                )

        avg_loss /= num_steps
        self.losses.append(avg_loss)
