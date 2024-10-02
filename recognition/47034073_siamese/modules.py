import logging
from dataclasses import dataclass

from torchvision.models import efficientnet_b0
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class HyperParams:
    num_epochs: int = 20
    batch_size: int = 32


class TumorClassifier:
    def __init__(self, hparams: HyperParams):
        self._model = TumorTower().to(device)
        self._optim = torch.optim.Adam(self._model.parameters())
        self._hparams = hparams
        self._losses = []
        self._benign_centroid = None
        self._malignant_centroid = None

    def train(self, train_loader: DataLoader):
        for epoch in range(self._hparams.num_epochs):
            self._train_epoch(train_loader)
            logger.debug("hello")
            logger.info("Epoch %d / loss %e", epoch, self._losses[-1])

    def compute_centroids(self, loader: DataLoader)
        self._benign_centroid = None
        self._malignant_centroid = None
        
        num_benign = 0
        num_malignant = 0
        with torch.inference_mode():
            for x, labels in loader:
                x = x.to(device)
                embeddings = self._model.compute_embedding(x)

                if self._benign_centroid == None:
                    embedding_dim = embeddings.shape[1]
                    self._benign_centroid = torch.zeros(embedding_dim)
                if self._malignant_centroid == None:
                    self._malignant_centroid = torch.zeros(embedding_dim)

                benign = embeddings[labels == 0]
                malignant = embeddings[labels == 1]
                
                num_benign += len(benign)
                num_malignant += len(malignant)

                self._benign_centroid += benign.sum(dim=0)
                self._malignant_centroid += malignant.sum(dim=0)

            self._benign_centroid /= num_benign
            self._malignant_centroid /= num_malignant

    def _train_epoch(self, train_loader):
        avg_loss = 0.0
        n = 0
        for x1, x2, y in train_loader:
            n += 1
            logger.debug("start train loop")
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.float().to(device)
            self._optim.zero_grad()
            logits = self._model(x1, x2)
            loss = binary_cross_entropy_with_logits(logits.flatten(), y)
            avg_loss += loss.item()
            loss.backward()
            self._optim.step()

        avg_loss /= n
        self._losses.append(avg_loss)



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
    
    def compute_embedding(self, x: torch.Tensor):
        return self._backbone(x)
