import numpy as np
import torch
from torch_geometric.data import Data

def load_facebook_data(filepath='facebook.npz'):
    data = np.load(filepath)

    # Use the 'target' key as the label
    features = torch.tensor(data['features'], dtype=torch.float)  # 128-dim node features
    edge_index = torch.tensor(data['edges'], dtype=torch.long)  # edge list
    labels = torch.tensor(data['target'], dtype=torch.long)  # Use 'target' as the label

    # Ensure edge_index is in the correct format
    edge_index = edge_index.t().contiguous()

    # Create a PyTorch Geometric data object
    data = Data(x=features, edge_index=edge_index, y=labels)

    return data
