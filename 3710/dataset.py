import numpy as np
import torch
from torch_geometric.data import Data

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

    # Generate training mask for semi-supervised learning
    num_nodes = len(labels)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    labeled_indices = np.random.choice(num_nodes, 1000, replace=False)
    train_mask[labeled_indices] = True

    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)

    # Print some information for debugging
    print("Features shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Labels shape:", y.shape)
    print("Data object:", data)
    return data
load_data('/Users/zhangxiangxu/Downloads/3710_report/facebook.npz')
