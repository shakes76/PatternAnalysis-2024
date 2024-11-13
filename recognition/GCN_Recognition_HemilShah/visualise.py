import umap
import matplotlib.pyplot as plt

def visualize(embeddings, graph_data):
    umap_embeddings = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='cosine').fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=graph_data.y.cpu(), cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('UMAP Embeddings of Facebook Network with Ground Truth Labels')
    plt.show()
