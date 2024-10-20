# Facebook Page-Page Network Node Classification

## Overview

This project involves training a Graph Neural Network (GNN) for semi-supervised multi-class node classification using the Facebook Large Page-Page Network dataset. The dataset represents various categories of Facebook pages (e.g., government, politician, tvshow, company) connected based on mutual likes. The task is to classify each node (Facebook page) into one of the provided categories using Graph Attention Networks (GAT) layers.

The model uses the Graph Attention Network (GAT) architecture and includes visualization of learned embeddings using UMAP.


## Dataset

The dataset is provided in the form of an partially processed .npz file containing:

Edges: Connections between nodes representing mutual page likes.

Features: 128-dimensional feature vectors extracted from page descriptions.

Target Labels: Page types (e.g., government, company, tvshow, politician).

## Model Architecture

The GNN model uses four layers of Graph Attention Networks (GATConv) with multi-head attention, combined with ReLU activations and dropout to prevent overfitting. The model architecture is as follows:

GATConv Layers:

- conv1: 8 attention heads for initial node feature transformation.

- conv2: 8 attention heads for further feature transformation.

- conv3: 8 attention heads for deeper representation learning.

- conv4: 1 attention head for the output layer.

Mixed precision training with automatic mixed precision (AMP) was utilized to speed up training and reduce GPU memory usage, enabling the use of larger hidden dimensions.

## Training Details

Hidden Dimension: 512.

Optimizer: AdamW with learning rate 0.0005 and weight decay 5e-4.

Loss Function: CrossEntropyLoss.

Epochs: 200.

Mixed Precision Training: Enabled using PyTorch AMP for increased training efficiency.

Training and Validation Loss

The following plot shows the training and validation loss over 200 epochs:



The training and validation loss decreased steadily, indicating that the model learned effectively without overfitting.
