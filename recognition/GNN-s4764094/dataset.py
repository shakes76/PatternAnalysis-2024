import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

# Upload our dataset and transfer to tensor format, return the edges, train and test set
def upload_dataset(device):
    """
    Loads the Facebook dataset which is partially preprocessed , which includes node features, edges, and targets.

    :param device: The device ('cpu', 'cuda', or 'mps') which is used for transferring tensor.
    :returns
        - tensor_edges: The edges of the graph, without self-loops.
        - train_set: The dataset for training, containing node features and their corresponding targets.
        - test_set: The dataset for testing, containing node features and their corresponding targets.
    """

    # Load the dataset from the provided file
    facebook_data = np.load("/Users/chenyihu/Desktop/Pycharm_Code/3710-PatternAnalysis-2024/facebook_large/facebook.npz")
    tensor_edges = torch.tensor(facebook_data['edges'].T).to(device)

    # Avoid self-looping
    tensor_edges = tensor_edges[:, tensor_edges[0] != tensor_edges[1]]
    tensor_targets = torch.tensor(facebook_data['target']).to(device)
    tensor_features = torch.tensor(facebook_data['features']).to(device)

    print("Nodes edges: ", tensor_edges)
    print("Nodes targets: ", tensor_targets)
    print("Nodes features: ", tensor_features)

    # Normalize the node features
    scaler = StandardScaler()
    tensor_features_cpu = tensor_features.cpu().numpy()
    normalized_features = scaler.fit_transform(tensor_features_cpu)
    tensor_features = torch.tensor(normalized_features, dtype=torch.float32).to(device)

    # Define the assignment of training and testing set
    num_nodes = tensor_targets.shape[0]
    node_indices = torch.arange(num_nodes)
    train_id, test_id = train_test_split(node_indices, test_size=0.7, random_state=42)

    train_features = tensor_features[train_id]
    train_targets = tensor_targets[train_id]

    test_features = tensor_features[test_id]
    test_targets = tensor_targets[test_id]

    train_set = TensorDataset(train_features, train_targets)
    test_set = TensorDataset(test_features, test_targets)
    return tensor_edges, train_set, test_set
