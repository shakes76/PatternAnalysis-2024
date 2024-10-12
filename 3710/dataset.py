import numpy as np
import torch
from torch_geometric.data import Data

def load_data(npz_file_path):
    # Step 1: Load the .npz data file
    data = np.load(npz_file_path)

    # Step 2: Extract features and edges from the data
    features = data['features']
    edges = data['edges']

    # Step 3: Convert features and edges into a PyTorch Geometric Data object
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)

    # Print some information for debugging
    print("Features shape:", features.shape)
    print("Edges shape:", edges.shape)
    print("Data object:", data)
    print("Feature min value:", features.min())
    print("Feature max value:", features.max())
    print("Feature mean:", features.mean())
    print("Feature standard deviation:", features.std())
    return data
load_data('/Users/zhangxiangxu/Downloads/3710_report/facebook.npz')
