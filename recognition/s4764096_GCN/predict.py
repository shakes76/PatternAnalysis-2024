import torch as th
import umap.umap_ as umap
import matplotlib.pyplot as plt
from dataset import prepare_data
from modules import GCN

def show_graph(graph, model):
    with th.no_grad():

        embeddings = model(graph, graph.ndata['features'])

    embeddings = embeddings.numpy()

    umap_embeddings = umap.UMAP(n_neighbors=30, min_dist=0.5, n_components=2).fit_transform(embeddings)
    true_labels = graph.ndata['labels'].numpy()

    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=true_labels, cmap='viridis', s=10)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    # Load data and model
    graph, _, _, in_feats = prepare_data()
    model = GCN(in_feats, num_classes=4)

    # Load pre-trained model weights here if available
    show_graph(graph, model)
