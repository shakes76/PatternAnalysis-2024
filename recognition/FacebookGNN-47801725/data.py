import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

def filter_edges(edge_index, num_nodes):
    """
    Filters out edges that reference nodes outside the valid range.

    Args:
        edge_index (torch.Tensor): The edge index tensor (2, num_edges).
        num_nodes (int): The number of nodes in the graph.

    Returns:
        torch.Tensor: Filtered edge index tensor.
    """
    # Create a mask to filter out edges where both source and target nodes are within the valid range
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    filtered_edge_index = edge_index[:, mask]
    return filtered_edge_index

def upload_dataset(file_path, device):
    """
    Loads and preprocesses the Facebook dataset.

    Args:
        file_path (str): Path to the .npz file containing the dataset.
        device (torch.device): The device to transfer tensors ('cpu', 'cuda', or 'mps').

    Returns:
        tensor_edges (torch.Tensor): The edges of the graph without self-loops.
        train_set (TensorDataset): The dataset for training, containing node features and corresponding targets.
        test_set (TensorDataset): The dataset for testing, containing node features and corresponding targets.
    """
    # Load the dataset from the provided file
    facebook_data = np.load(file_path)
    edges = facebook_data['edges']
    features = facebook_data['features']
    targets = facebook_data['target']

    # Convert edges to tensor and remove self-loops
    tensor_edges = torch.tensor(edges.T, dtype=torch.long).to(device)
    tensor_edges = tensor_edges[:, tensor_edges[0] != tensor_edges[1]]  # Remove self-loops

    # Convert features and targets to tensors
    tensor_features = torch.tensor(features, dtype=torch.float32).to(device)
    tensor_targets = torch.tensor(targets, dtype=torch.long).to(device)

    # Normalize the features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(tensor_features.cpu().numpy())
    tensor_features = torch.tensor(normalized_features, dtype=torch.float32).to(device)

    # Filter out edges that reference non-existing nodes
    num_nodes = tensor_features.size(0)
    tensor_edges = filter_edges(tensor_edges, num_nodes)

    # Split into train and test datasets
    node_indices = torch.arange(num_nodes)
    train_idx, test_idx = train_test_split(node_indices, test_size=0.7, random_state=42)

    train_features = tensor_features[train_idx]
    train_targets = tensor_targets[train_idx]
    test_features = tensor_features[test_idx]
    test_targets = tensor_targets[test_idx]

    # Create TensorDatasets for training and testing
    train_set = TensorDataset(train_features, train_targets)
    test_set = TensorDataset(test_features, test_targets)

    return tensor_edges, train_set, test_set



