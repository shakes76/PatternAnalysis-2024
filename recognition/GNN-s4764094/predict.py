from sklearn.manifold import TSNE
import numpy as np
import torch
import matplotlib.pyplot as plt

def extract_embeddings(model, loader, edges, device):
    """
    Extracts embeddings from the model.

    :param model: Our trained / untrained GCN model.
    :param loader: DataLoader object which contains the dataset.
    :param edges: The graph edges for GCN model.
    :param device: CUDA / MPS device to run the program.
    :return: Concatenated embeddings and their corresponding labels.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            embeddings = model.get_embeddings(features, edges)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)

# Plot t-SNE of embeddings with custom point size and colors
def plot_tsne(embeddings, labels, title="t-SNE Embeddings"):
    """
    Implementing TSNE for dimensionality reduction, visualizing the embeddings' performance.
    Reference: The knowledge of 2D t-SNE algorithm is learnt and referred from
    https://www.datacamp.com/tutorial/introduction-t-sne.

    :param embeddings: Relevant embeddings that are extracted.
    :param labels: The corresponding labels for the data points.
    :param title: Title of the plot.
    :return: The 2D t-SNE plot with different classes in specified colors.
    """
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_transform = tsne.fit_transform(embeddings)

    colors = ['Green', 'Blue', 'Red', 'Yellow']

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings_transform[:, 0], embeddings_transform[:, 1],
                          c=labels, cmap=plt.matplotlib.colors.ListedColormap(colors), alpha=0.5, s=8)
    plt.colorbar(scatter, ticks=[0, 1, 2, 3], label='Classes')
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
