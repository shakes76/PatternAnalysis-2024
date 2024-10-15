import pathlib
import logging

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn import preprocessing

from trainer import SiameseController

PLOTS_PATH = pathlib.Path("plots")

logger = logging.getLogger(__name__)


def plot_training_data(trainer: SiameseController) -> None:
    """Save training plots using data from trainer."""
    plt.figure()
    plt.plot(trainer.losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "train_loss")
    plt.figure()
    plt.plot(trainer.mined_each_step)
    plt.xlabel("Train step")
    plt.ylabel("Num mined")
    plt.tight_layout()
    plt.savefig(PLOTS_PATH / "mined")


def plot_pca(embeddings, targets) -> None:
    """Save plot of embeddings in pca space."""
    logger.info("Fitting pca...")
    standard_embeddings = preprocessing.scale(embeddings)
    pca = PCA(n_components=2)
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
