# ====================================================
# File: visualise.py
# Description: Provides functions for visualizing data, model performance, and predictions.
# Author: Hemil Shah
# Date Created: 14-11-2024
# Version: 1.0
# License: MIT License
# ====================================================

import umap
import matplotlib.pyplot as plt

def visualize(embeddings, graph_data):
    umap_embeddings = umap.UMAP(n_neighbors=10, min_dist=0.3, metric='cosine').fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=graph_data.y.cpu(), cmap='Spectral', s=5)
    plt.colorbar()
    plt.title('UMAP Embeddings of Facebook Network with Ground Truth Labels')
    plt.show()
