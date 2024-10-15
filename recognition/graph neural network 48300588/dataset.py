import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data():
    # Load the npz file
    data_npz = np.load('e:/xuexi/UQ/2024.s2/COMP 3710/A3/facebook.npz')
    
    # Extract node features
    x = torch.tensor(data_npz['features'], dtype=torch.float)
    print('x.shape:', x.shape)  # Debug output

    # Extract edge information
    edge_index = torch.tensor(data_npz['edges'], dtype=torch.long)
    print('Original edge_index.shape:', edge_index.shape)  # Debug output

    # Check the shape of edge_index, if it's [num_edges, 2], transpose it
    if edge_index.shape[0] == 2:
        # Shape is already [2, num_edges]
        edge_index = edge_index
    else:
        # Transpose to match [2, num_edges]
        edge_index = edge_index.T
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    print('Adjusted edge_index.shape:', edge_index.shape)  # Debug output

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
    labeled_indices = np.arange(num_nodes)
    np.random.shuffle(labeled_indices)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    train_indices = labeled_indices[:train_size]
    val_indices = labeled_indices[train_size:train_size+val_size]
    test_indices = labeled_indices[train_size+val_size:]

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return data, label_encoder.classes_
