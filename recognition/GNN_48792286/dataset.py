# Import packages
import numpy as np
import torch
from torch_geometric.data import Data

# Function to load and preprocess the data
def load_data(file_path):
    # Load the dataset from a .npz file
    data = np.load(file_path)

    # Extract features, edges, and target labels
    features = data['features']
    edges = data['edges']
    targets = data['target']

    # Ensure targets are a one-dimensional array
    if targets.ndim == 1:
        labels = targets
    else:
        labels = targets[:, 0]  # If two-dimensional, extract the first column

    # Convert features, edges, and labels into PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create a graph data object
    graph_data = Data(x=x, edge_index=edge_index, y=y)

    # Create a training mask (80% of nodes for training)
    num_nodes = graph_data.x.size(0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(num_nodes * 0.8)] = True
    graph_data.train_mask = train_mask

    return graph_data
