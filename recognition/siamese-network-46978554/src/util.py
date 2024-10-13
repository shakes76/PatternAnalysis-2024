from pathlib import Path

# Dataset directory
# We use a downsized version of the ISIC 2020 dataset:
# https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data
DATA_DIR = Path(__file__).parent.parent / "data"


def plot_pca_embeddings(embeddings, labels, pred=False):
    """
    Plots the first two principal components of the given embeddings.

    Args:
        embeddings: Model output embeddings.
        labels: Labels for each embedding. Used for colouring the embedded points.
        pred: Whether the labels are predictions or targets (i.e. ground truth).
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    comp1 = embeddings_pca[:, 0]
    comp2 = embeddings_pca[:, 1]

    suffix = "pred" if pred else "target"

    plt.scatter(comp1[labels == 0], comp2[labels == 0], label="benign " + suffix)
    plt.scatter(comp1[labels == 1], comp2[labels == 1], label="malignant " + suffix)
    plt.xlabel("PC Direction 1")
    plt.ylabel("PC Direction 2")
    plt.title("Model Embeddings (Test Set) in Principal Component Space")
    plt.legend()
    plt.show()


def plot_tsne_embeddings(embeddings, labels, pred=False):
    """
    Plots the first two TSNE components of the given embeddings.

    Args:
        embeddings: Model output embeddings.
        labels: Labels for each embedding. Used for colouring the embedded points.
        pred: Whether the labels are predictions or targets (i.e. ground truth).
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(embeddings)

    comp1 = embeddings_tsne[:, 0]
    comp2 = embeddings_tsne[:, 1]

    suffix = "pred" if pred else "target"

    plt.scatter(comp1[labels == 0], comp2[labels == 0], label="benign " + suffix)
    plt.scatter(comp1[labels == 1], comp2[labels == 1], label="malignant " + suffix)
    plt.xlabel("TSNE Direction 1")
    plt.ylabel("TSNE Direction 2")
    plt.title("Model Embeddings (Test Set) in Principal Component Space")
    plt.legend()
    plt.show()


def plot_losses(loss_files, loss_names):
    import matplotlib.pyplot as plt
    import torch

    # Average every 100 losses
    skip = 100

    for loss_file, loss_name in zip(loss_files, loss_names):
        loss = torch.load(loss_file, weights_only=False)
        loss = loss[:len(loss) // skip * skip]
        loss = loss.view(len(loss) // skip, skip)
        mean_loss = loss.mean(dim=1)
        plt.plot(mean_loss, label=loss_name)

    plt.xlabel("Epoch mini batches (averaged every 100)")
    plt.ylabel("Loss")
    plt.title("Loss over epochs (averaged every 100)")
    plt.legend()
    plt.show()

