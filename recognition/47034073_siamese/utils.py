import pathlib

import matplotlib.pyplot as plt

from trainer import SiameseController

PLOTS_PATH = pathlib.Path("plots")


def save_training_plots(trainer: SiameseController):
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
