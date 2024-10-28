"""
File: utils.py
Description: Utility functions used for model implementation.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch
import dataset
import modules
import train
import os
import csv

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

def display_gnn_tsne() -> None:
    """
        Display the t-distributed stochastic neighbour embeddings for
        the trained Facebook Page-Page Network model using the dataset categories.
    """
    _, flpp_data, _, _, _ = dataset.load_dataset(train.DATASET_DIR)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("WARNING: CUDA not found; Using CPU")

    model = modules.GNN(128, 16, 4)
    train._load_model(model, 'gnn_classifier')

    model.to(device)
    model.eval()

    # Turn off gradient descent when we run inference on the model
    with torch.no_grad():

        # Get the predicted classes for this batch
        outputs = model(flpp_data.x, flpp_data.edge_index)

    model_tsne = TSNE(n_components=2, perplexity=30, random_state=1)
    reduced_embeddings = model_tsne.fit_transform(outputs.cpu().numpy())

    plt.figure(figsize=(10, 10))

    for i, category in enumerate(dataset.FLPP_CATEGORIES):
        idx = flpp_data.y == i
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=category, alpha = 0.5)

    plt.legend()
    plt.title("GNN TSNE Plot")
    plt.show()

def display_raw_tsne() -> None:
    """
        Display the t-distributed stochastic neighbour embeddings for
        the untrained Facebook Page-Page Network model using the dataset categories.
    """
    _, flpp_data, _, _, _ = dataset.load_dataset(train.DATASET_DIR)

    model_tsne = TSNE(n_components=2, perplexity=30, random_state=1)
    reduced_embeddings = model_tsne.fit_transform(flpp_data.x)

    plt.figure(figsize=(10, 10))

    for i, category in enumerate(dataset.FLPP_CATEGORIES):
        idx = flpp_data.y == i
        plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=category, alpha = 0.5)

    plt.legend()
    plt.title("Raw TSNE Plot")
    plt.show()

def display_training():
    """
        Display save training data from disk using matplot lib
    """
    csv_path = os.path.join(train.CSV_DIR, f'gnn_classifier.csv')

    epoch_nums = []
    training_losses = []
    validation_losses = []
    validation_accuracies = []

    # Open saved csv data for model
    with open(csv_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)

        for i, row in enumerate(csv_reader):
            _, training_loss, validation_loss, validation_accuracy, duration = row

            epoch_nums.append(i)
            training_losses.append(float(training_loss))
            validation_losses.append(float(validation_loss))
            validation_accuracies.append(float(validation_accuracy))

    _, ax1 = plt.subplots()

    # Create plot from saved CSV data
    plt.title("GNN Epoch Accuracy and Loss")

    ax1.plot(epoch_nums, training_losses)
    ax1.plot(epoch_nums, validation_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(['Training', 'Validation'], loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(epoch_nums, validation_accuracies, 'g-')
    ax2.set_ylabel('Accuracy')
    ax2.legend(['Accuracy'], loc='upper right')

    plt.show()
