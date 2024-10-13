from pathlib import Path

# Dataset directory
# We use a downsized version of the ISIC 2020 dataset:
# https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data
DATA_DIR = Path(__file__).parent.parent / "data"


def plot_pca_embeddings(embeddings, targets):
    """
    Plots the first two principal components of the given embeddings.

    Args:
        embeddings: Model output embeddings.
        targets: Labels for each embedding. Used for colouring the embedded points.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    comp1 = embeddings_pca[:, 0]
    comp2 = embeddings_pca[:, 1]

    plt.scatter(comp1[targets == 0], comp2[targets == 0], label="benign")
    plt.scatter(comp1[targets == 1], comp2[targets == 1], label="malignant")
    plt.xlabel("PC Direction 1")
    plt.ylabel("PC Direction 2")
    plt.title("Model Embeddings (Test Set) in Principal Component Space")
    plt.legend()
    plt.show()


def plot_tsne_embeddings(embeddings, targets):
    """
    Plots the first two TSNE components of the given embeddings.

    Args:
        embeddings: Model output embeddings.
        targets: Labels for each embedding. Used for colouring the embedded points.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(embeddings)

    comp1 = embeddings_tsne[:, 0]
    comp2 = embeddings_tsne[:, 1]

    plt.scatter(comp1[targets == 0], comp2[targets == 0], label="benign")
    plt.scatter(comp1[targets == 1], comp2[targets == 1], label="malignant")
    plt.xlabel("TSNE Direction 1")
    plt.ylabel("TSNE Direction 2")
    plt.title("Model Embeddings (Test Set) in Principal Component Space")
    plt.legend()
    plt.show()
