import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch_geometric.utils import add_self_loops

def load_data():
    # Load the npz file
    data_npz = np.load('e:/xuexi/UQ/2024.s2/COMP 3710/A3/facebook.npz')
    
    # Extract node features
    x = torch.tensor(data_npz['features'], dtype=torch.float)
    print('x.shape:', x.shape)  # Debug output

    # Extract edge information
    edges = data_npz['edges']
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print('edge_index.shape:', edge_index.shape)  # Debug output

    # Add self-loops to the adjacency matrix
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # Extract labels
    y = data_npz['target']
    print('y.shape:', y.shape)  # Debug output

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = torch.tensor(y, dtype=torch.long)

    # Create masks
    num_nodes = x.shape[0]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Split dataset
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    test_size = num_nodes - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    print(f"Number of training nodes: {train_mask.sum().item()}")
    print(f"Number of validation nodes: {val_mask.sum().item()}")
    print(f"Number of test nodes: {test_mask.sum().item()}")

    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data, label_encoder.classes_
