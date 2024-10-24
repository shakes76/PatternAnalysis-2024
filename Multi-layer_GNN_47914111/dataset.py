"""
Author: Yucheng Wang
Student ID: 47914111
This script implements a complete data loading and preprocessing pipeline for 
the Facebook Large Page-Page Network dataset. It includes functions for:

1. Loading data from a file and converting it to PyTorch tensors.
2. Extracting relevant data information (number of nodes, features, and classes).
3. Creating training, validation, and test masks.
4. Preprocessing the adjacency matrix by adding self-loops, normalizing it, and 
   converting it to a sparse PyTorch tensor.

The final processed data includes the normalized adjacency matrix, node features, 
node labels, and masks for training, validation, and testing.
"""
import numpy as np
import torch
import scipy.sparse as sp


def load_data():
    """
    Load data from a fixed file path and convert it to tensors.
    """
    filepath = "C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/facebook_large/facebook.npz"
    data = np.load(filepath)

    # Extract numpy arrays
    edges = data["edges"]
    features = data["features"]
    labels = data["target"]

    # Convert numpy arrays to PyTorch tensors
    edge_list = torch.tensor(edges, dtype=torch.long)
    feature_matrix = torch.tensor(features, dtype=torch.float)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    # Print the shapes of the tensors
    print(f"Edges Shape: {edge_list.shape}")
    print(f"Feature Matrix Shape: {feature_matrix.shape}")
    print(f"Labels Shape: {label_tensor.shape}")

    return edge_list, feature_matrix, label_tensor


def generate_data_info():
    """
    Get the number of nodes, features, and classes in the data.
    """
    # Call load_data to load the data
    edge_list, feature_matrix, label_tensor = load_data()

    num_nodes = feature_matrix.size(0)
    num_features = feature_matrix.size(1)
    num_classes = len(torch.unique(label_tensor))

    # Print data information
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Features: {num_features}")
    print(f"Number of Classes: {num_classes}")

    # Return the data information
    return num_nodes, num_features, num_classes, edge_list, feature_matrix, label_tensor


def generate_data_split():
    """
    Create masks for training, validation, and test splits.
    """
    # Get data information and edge_list
    num_nodes, _, _, edge_list, feature_matrix, label_tensor = generate_data_info()

    # Set the ratios for training and validation sets
    train_ratio = 0.7
    validation_ratio = 0.15

    # Calculate the size of each data split
    train_size = int(train_ratio * num_nodes)
    validation_size = int(validation_ratio * num_nodes)

    # Shuffle the node indices
    shuffled_indices = torch.randperm(num_nodes)

    # Create masks for each dataset
    train_indices = shuffled_indices[:train_size]
    validation_indices = shuffled_indices[train_size : train_size + validation_size]
    test_indices = shuffled_indices[train_size + validation_size :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
    validation_mask[validation_indices] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    # Print the size of each dataset
    print(f"Training Set Size: {train_mask.sum().item()}")
    print(f"Validation Set Size: {validation_mask.sum().item()}")
    print(f"Test Set Size: {test_mask.sum().item()}")

    return train_mask, validation_mask, test_mask, edge_list, feature_matrix, label_tensor


def preprocess_adjacency_matrix():
    """
    Add self-loops, normalize the adjacency matrix, and convert it to a sparse tensor.
    """
    # Get masks and data information
    train_mask, validation_mask, test_mask, edge_list, feature_matrix, label_tensor = generate_data_split()

    num_nodes = feature_matrix.size(0)

    # Create sparse adjacency matrix
    adjacency_matrix = sp.coo_matrix(
        (np.ones(edge_list.shape[0]), (edge_list[:, 0], edge_list[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    # Add self-loops
    adjacency_matrix += sp.eye(num_nodes)

    # Normalize the adjacency matrix
    row_sum = np.array(adjacency_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adjacency_matrix = adjacency_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    # Convert to PyTorch sparse tensor
    adjacency_indices = torch.tensor(np.vstack((normalized_adjacency_matrix.row, normalized_adjacency_matrix.col)), dtype=torch.long)
    adjacency_values = torch.tensor(normalized_adjacency_matrix.data, dtype=torch.float)

    normalized_adjacency_tensor = torch.sparse_coo_tensor(adjacency_indices, adjacency_values, torch.Size([num_nodes, num_nodes]))

    # Print adjacency matrix information
    print(f"Normalized Adjacency Matrix Shape: {normalized_adjacency_tensor.shape}")
    print(f"Number of Non-Zero Elements in Adjacency Matrix: {normalized_adjacency_tensor._nnz()}")

    # Print complete data processing information
    print("Data preprocessing complete.")
    print(f"Feature Matrix Shape: {feature_matrix.shape}")
    print(f"Labels Shape: {label_tensor.shape}")
    print(f"Train Mask: {train_mask.sum().item()}")
    print(f"Validation Mask: {validation_mask.sum().item()}")
    print(f"Test Mask: {test_mask.sum().item()}")

    return normalized_adjacency_tensor, train_mask, validation_mask, test_mask, feature_matrix, label_tensor

preprocess_adjacency_matrix()
