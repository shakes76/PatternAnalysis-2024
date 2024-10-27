"""
File: utils.py
Description: Utility functions used for model implementation.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import dataset
import modules

import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import FacebookPagePage
from torch_geometric.utils.convert import to_networkx

from sklearn.manifold import TSNE

def display_flpp_network(flpp_dataset: FacebookPagePage) -> None:
    """
        Display the FLPP network connections via edges.

        Parameters:
            flpp_dataset: The Facebook Page-Page Network data set
    """
    flpp_graph_nx = to_networkx(flpp_dataset[0])

    fig, ax = plt.subplots(figsize=(15, 9))
    pos = nx.spring_layout(flpp_graph_nx, iterations=15, seed=1721)

    ax.axis("off")

    plot_options = {"node_size": 10, "with_labels": False, "width": 0.15}
    nx.draw_networkx(flpp_graph_nx, pos=pos, ax=ax, **plot_options)

    plt.show()

def display_gnn_tsne(model: modules.GNN, flpp_dataset: FacebookPagePage) -> None:
    """
        Display the t-distributed stochastic neighbour embeddings for
        the trained Facebook Page-Page Network model using the dataset categories.

        Parameters:
            model: The trained model used for the t-distributed stochastic neighbour embedding.
            flpp_dataset: The raw dataset used for categorisation.
    """
    model_tsne = TSNE(n_components=2, perplexity=30, random_state=1)
    reduced_embeddings = model_tsne.fit_transform(model.cpu().numpy())

    plt.figure(figsize=(10, 10))

    for i, category in enumerate(dataset.FLPP_CATEGORIES):
        idx = flpp_dataset.y == i
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=category, alpha = 0.5)

    plt.legend()
    plt.title("GNN TSNE Plot")
    plt.show()

    #plt.savefig("GG_TSNE_plot.png")

def display_raw_tsne(flpp_dataset: FacebookPagePage) -> None:
    """
        Display the t-distributed stochastic neighbour embeddings for
        the untrained Facebook Page-Page Network model using the dataset categories.

        Parameters:
            flpp_dataset: The raw dataset used for categorisation.
    """
    model_tsne = TSNE(n_components=2, perplexity=30, random_state=1)
    reduced_embeddings = model_tsne.fit_transform(flpp_dataset.x)

    plt.figure(figsize=(10, 10))

    for i, category in enumerate(dataset.FLPP_CATEGORIES):
        idx = flpp_dataset.y == i
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=category, alpha = 0.5)

    plt.legend()
    plt.title("Raw TSNE Plot")
    plt.show()

    #plt.savefig("GG_TSNE_plot.png")
