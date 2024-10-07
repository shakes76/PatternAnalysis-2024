from pathlib import Path

import torch
import torch.nn.functional as F

# Dataset directory
# We use a downsized version of the ISIC 2020 dataset:
# https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data
DATA_DIR = Path(__file__).parent.parent / "data"


def contrastive_loss(margin):
    """
    REF: https://www.sciencedirect.com/topics/computer-science/contrastive-loss
    """

    def f(x1, x2, y):
        dist = F.pairwise_distance(x1, x2)
        dist_sq = torch.pow(dist, 2)

        loss = (1 - y) * dist_sq + y * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
        loss = torch.mean(loss / 2.0, dim=0)

        return loss

    return f


def contrastive_loss_threshold(margin):
    def f(x1, x2):
        dist = F.pairwise_distance(x1, x2)
        return (dist >= margin).float()

    return f
