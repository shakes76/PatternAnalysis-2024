import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

def create_masks(labels, num_labeled=800, train_ratio=0.7, val_ratio=0.15):
    # Select indices of labeled nodes
    labeled_indices = (labels != -1).nonzero(as_tuple=True)[0].numpy()  # Assuming -1 represents no label
    np.random.shuffle(labeled_indices)

    # Split labeled nodes into train, validation, and test sets
    num_labeled = min(num_labeled, len(labeled_indices))
    labeled_indices = labeled_indices[:num_labeled]
    
    train_split = int(train_ratio * num_labeled)
    val_split = int(val_ratio * num_labeled)

    train_indices = labeled_indices[:train_split]
    val_indices = labeled_indices[train_split:train_split + val_split]
    test_indices = labeled_indices[train_split + val_split:]

    # Create masks for train, validation, and test sets
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

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

    # Step 3: Convert features, edges, and labels into a PyTorch Geometric Data object
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    # Add self-loops
    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    # Generate train, validation, and test masks only for labeled nodes
    train_mask, val_mask, test_mask = create_masks(y)

    # Create Data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # Print some debugging information
    print("Features shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Labels shape:", y.shape)
    print("Number of labeled nodes:", (y != -1).sum().item())
    print("Number of training nodes:", train_mask.sum().item())
    print("Number of validation nodes:", val_mask.sum().item())
    print("Number of test nodes:", test_mask.sum().item())
    print("Data object:", data)

    return data

