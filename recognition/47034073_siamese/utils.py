"""Visualisation and metric computing utilities"""

import pathlib
import logging

import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing

from trainer import SiameseController
from dataset import LesionClassificationDataset

PLOTS_PATH = pathlib.Path("plots")
IMAGES_PATH = pathlib.Path("data/small_images")

logger = logging.getLogger(__name__)


def plot_training_data(trainer: SiameseController) -> None:
    """Save training plots using data from trainer."""
    plt.figure()
    plt.plot(trainer.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "train_loss")
    plt.figure(figsize=(20, 10))
    plt.plot(trainer.mined_each_step, linewidth=0.5)
    plt.xlabel("Train step")
    plt.ylabel("Num mined")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "mined")


def plot_pca(embeddings, targets) -> None:
    """Save plot of embeddings in pca space."""
    logger.info("Fitting pca...")
    standard_embeddings = preprocessing.scale(embeddings)
    pca = PCA(n_components=2, random_state=42)
    pca_projections = pca.fit_transform(standard_embeddings)
    logger.info("Plotting pca...")
    plt.figure()
    plt.scatter(
        pca_projections[:, 0],
        pca_projections[:, 1],
        c=targets,
        cmap="coolwarm",
        marker=".",
        s=0.5,
    )
    plt.xlabel("component1")
    plt.ylabel("component2")
    benign_patch = mpatches.Patch(color="blue", label="Benign")
    malignant_patch = mpatches.Patch(color="red", label="Malignant")
    plt.legend(handles=[benign_patch, malignant_patch])
    logger.info("Writing image")
    plt.savefig(PLOTS_PATH / "train_pca")


def plot_tsne(embeddings, targets) -> None:
    """Save plot of embeddings in tsne space."""
    logger.info("Fitting tsne...")
    standard_embeddings = preprocessing.scale(embeddings)
    tsne = TSNE(random_state=42)
    tsne_projections = tsne.fit_transform(standard_embeddings)
    logger.info("Plotting tsne...")
    plt.figure()
    plt.scatter(
        tsne_projections[:, 0],
        tsne_projections[:, 1],
        c=targets,
        cmap="coolwarm",
        marker=".",
    )
    plt.xlabel("component1")
    plt.ylabel("component2")
    benign_patch = mpatches.Patch(color="blue", label="Benign")
    malignant_patch = mpatches.Patch(color="red", label="Malignant")
    plt.legend(handles=[benign_patch, malignant_patch])
    logger.info("Writing image")
    plt.savefig(PLOTS_PATH / "train_tsne")


def get_classifier_loader(
    train_meta_df: pd.DataFrame, batch_size: int = 128, num_workers: int = 2
) -> tuple[pd.DataFrame, DataLoader]:
    """Return loader for classifier"""
    benign = train_meta_df[train_meta_df["target"] == 0]
    malignant = train_meta_df[train_meta_df["target"] == 1]
    benign = benign.sample(random_state=42, n=len(malignant))
    balanced_df = pd.concat([benign, malignant])
    balanced_ds = LesionClassificationDataset(IMAGES_PATH, balanced_df, transform=False)
    loader = DataLoader(
        balanced_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return balanced_df, loader
