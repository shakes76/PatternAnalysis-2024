import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

def load_dataset(file_path, device):
    """
    Load and preprocess the Facebook dataset.
    
    Args:
        file_path (str): Path to the .npz file containing the dataset.
        device (torch.device): The device to move tensors ('cpu' or 'cuda').
        
    Returns:
        tensor_edges (torch.Tensor): The graph edges without self-loops.
        train_set (TensorDataset): Training dataset with features and targets.
        test_set (TensorDataset): Testing dataset with features and targets.
    """
    # Load dataset
    data = np.load(file_path)
    edges = data['edges']
    features = data['features']
    targets = data['target']
    
    # Preprocess features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32).to(device)
    targets = torch.tensor(targets, dtype=torch.long).to(device)
    
    # Convert edges to tensor and remove self-loops
    edges = torch.tensor(edges.T, dtype=torch.long).to(device)
    edges = edges[:, edges[0] != edges[1]]
    
    # Split into train and test
    num_nodes = targets.size(0)
    node_indices = torch.arange(num_nodes)
    train_idx, test_idx = train_test_split(node_indices, test_size=0.7, random_state=42)
    
    # Create training and testing sets
    train_features = features[train_idx]
    train_targets = targets[train_idx]
    test_features = features[test_idx]
    test_targets = targets[test_idx]
    
    train_set = TensorDataset(train_features, train_targets)
    test_set = TensorDataset(test_features, test_targets)
    
    return edges, train_set, test_set
