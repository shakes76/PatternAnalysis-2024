# Multi-Layer Graph Neural Network For Categorisation of Facebook Large Page-Page Network Dataset
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024

# Project Specification

Creates a multi-layer graph neural network (GNN) model for semi supervised multi-class node classification using Facebook Large Page-Page Network dataset with 80% Accuracy. This report includes TSNE and UMAP embeddings plot before and after model training providing a brief interpretation and discussion of results.

# Dependencies

- `python 3.11`
- `pytorch 2.4`
- `pytorch geometric 2.4`
- `matplotlib`
- `numpy`
- `scipy`
- `networkx`

# Data

The [Facebook Large Page-Page Network](https://snap.stanford.edu/data/facebook-large-page-page-network.html) dataset is utilised where nodes represent official Facebook pages while the links are mutual likes between sites. The dataset was pre-processed into a 3 numpy arrays:
- edges: The undirected graph connections between nodes in the dataset formatted as tuples from the start node to the end node.
- features: The features of each of the nodes stored in a 128 dimensions vector.
- targets: The categorisation target of each node.
The dataset has 22470 nodes with a 128 dimension feature set,  sourced from [Graph Mining Datasets](https://graphmining.ai/datasets/ptg/facebook.npz).


