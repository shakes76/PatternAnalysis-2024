import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Upload our dataset and transfer to tensor format, return those parameters
def upload_dataset(device):
    data = np.load("/Users/chenyihu/Desktop/Pycharm_Code/3710-PatternAnalysis-2024/facebook_large/facebook.npz")
    tensor_edges = torch.tensor(data['edges'].T).to(device)

    tensor_edges = tensor_edges[:, tensor_edges[0] != tensor_edges[1]]
    tensor_targets = torch.tensor(data['target']).to(device)
    tensor_features = torch.tensor(data['features']).to(device)
    print("Nodes edges: ", tensor_edges)
    print("Nodes targets: ", tensor_targets)
    print("Nodes features: ", tensor_features)

    # Define the assignment of training, testing and CV set
    num_nodes = tensor_targets.shape[0]
    node_indices = torch.arange(num_nodes)
    train_id, test_id = train_test_split(node_indices, test_size=0.8, random_state=42)

    train_features = tensor_features[train_id]
    train_targets = tensor_targets[train_id]

    test_features = tensor_features[test_id]
    test_targets = tensor_targets[test_id]

    train_set = TensorDataset(train_features, train_targets)
    test_set = TensorDataset(test_features, test_targets)
    return tensor_edges, train_set, test_set

