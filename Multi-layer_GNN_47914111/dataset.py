import numpy as np
import torch
import scipy.sparse as sp


def load_and_preprocess_data(filepath, train_ratio=0.7, validation_ratio=0.15):
    """
    Load data from .npz file, process it for GCN, and split into training, validation, and test sets.
    """
    # Load data file
    filepath = "C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/facebook_large/facebook.npz"
    facebook_data = np.load(filepath)

    # Extract individual numpy arrays
    edge_list = facebook_data["edges"]
    feature_matrix = facebook_data["features"]
    labels = facebook_data["target"]

    # Convert Numpy Arrays to Tensors
    edge_list_tensor = torch.tensor(edge_list, dtype=torch.long)
    feature_matrix_tensor = torch.tensor(feature_matrix, dtype=torch.float)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    # Print Shape of Tensors
    print(f"Edge List Shape: {edge_list_tensor.shape}")
    print(f"Feature Matrix Shape: {feature_matrix_tensor.shape}")
    print(f"Label Shape: {label_tensor.shape}")

    # Data Info
    num_nodes = feature_matrix_tensor.size(0)
    num_features = feature_matrix_tensor.size(1)
    num_classes = torch.unique(label_tensor).size(0)

    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Features: {num_features}")
    print(f"Number of Classes: {num_classes}")

    # Data Split
    train_size = int(train_ratio * num_nodes)
    validation_size = int(validation_ratio * num_nodes)

    shuffled_indices = torch.randperm(num_nodes)

    train_indices = shuffled_indices[:train_size]
    validation_indices = shuffled_indices[train_size : train_size + validation_size]
    test_indices = shuffled_indices[train_size + validation_size :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
    validation_mask[validation_indices] = True

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_indices] = True

    print(f"Training Set Size: {train_mask.sum().item()}")
    print(f"Validation Set Size: {validation_mask.sum().item()}")
    print(f"Test Set Size: {test_mask.sum().item()}")

    # Normalize adjacency matrix for GCN
    adjacency_matrix = sp.coo_matrix(
        (np.ones(edge_list_tensor.shape[0]), (edge_list_tensor[:, 0], edge_list_tensor[:, 1])),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )

    # Add self-loops
    adjacency_matrix += sp.eye(num_nodes)

    # Normalize adjacency matrix
    row_sum = np.array(adjacency_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    degree_matrix_inv_sqrt = sp.diags(d_inv_sqrt)

    normalized_adjacency_matrix = adjacency_matrix.dot(degree_matrix_inv_sqrt).transpose().dot(degree_matrix_inv_sqrt).tocoo()

    # Convert sparse matrix to torch sparse tensor
    adjacency_indices = torch.tensor(np.vstack((normalized_adjacency_matrix.row, normalized_adjacency_matrix.col)), dtype=torch.long)
    adjacency_values = torch.tensor(normalized_adjacency_matrix.data, dtype=torch.float)

    normalized_adjacency_tensor = torch.sparse_coo_tensor(adjacency_indices, adjacency_values, torch.Size([num_nodes, num_nodes]))

    # Output normalized adjacency matrix information
    print(f"Normalized Adjacency Matrix Shape: {normalized_adjacency_tensor.shape}")
    print(f"Number of Non-Zero Elements in Adjacency Matrix: {normalized_adjacency_tensor._nnz()}")

    return normalized_adjacency_tensor, feature_matrix_tensor, label_tensor, train_mask, validation_mask, test_mask


# Call the function with the updated filepath and return more meaningful variables
adjacency_matrix, feature_matrix, labels, train_mask, validation_mask, test_mask = load_and_preprocess_data("C:/Users/Wangyucheng/Desktop/comp3710a3/PatternAnalysis-2024/facebook_large/facebook.npz")
