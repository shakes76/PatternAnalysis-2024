"""
Author: Xiangxu Zhang
student number: 47643439
This script is designed to handle the loading, preprocessing, 
and splitting of the data in the facebook.npz file into several different collections for subsequent use.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

def create_masks(labels, train_ratio=0.7, val_ratio=0.15):
    # Determine the total number of nodes
    num_nodes = len(labels)
    # Create an array of indices from 0 to num_nodes - 1
    indices = np.arange(num_nodes)
    # Shuffle the indices to randomize the data
    np.random.shuffle(indices)

    # Calculate the number of nodes in each split (train, validation, test)
    train_split = int(train_ratio * num_nodes)
    val_split = int(val_ratio * num_nodes)

    # Assign indices to the respective data splits
    train_indices = indices[:train_split]
    val_indices = indices[train_split:train_split + val_split]
    test_indices = indices[train_split + val_split:]

    # Create boolean masks for training, validation, and testing sets
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Set the mask values for the respective indices to True
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    return train_mask, val_mask, test_mask

def load_data(npz_file_path):
    # Step 1: Load the .npz data file
    data = np.load(npz_file_path)

    # Step 2: Extract features, edges, and labels from the data
    features = data['features']
    edges = data['edges']
    labels = data['target']

    # Step 3: Convert features, edges, and labels to PyTorch Geometric's Data object format
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Convert edge list to PyTorch tensor
    x = torch.tensor(features, dtype=torch.float)  # Convert features to PyTorch tensor
    y = torch.tensor(labels, dtype=torch.long)  # Convert labels to PyTorch tensor

    # Add self-loops to the edge index (each node will have an edge to itself)
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # Randomly split all nodes into training, validation, and test sets
    train_mask, val_mask, test_mask = create_masks(y)

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # Print some debugging information
    print("Features shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Labels shape:", y.shape)
    print("Number of nodes:", x.size(0))
    print("Number of training nodes:", train_mask.sum().item())
    print("Number of validation nodes:", val_mask.sum().item())
    print("Number of test nodes:", test_mask.sum().item())
    print("Data object:", data)

    return data

