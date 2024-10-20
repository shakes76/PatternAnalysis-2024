"""
The data loader for the GNN node classification model
@author Anthony Ngo: s4801902
@date 21/10/2024
"""

import numpy as np
import torch
from torch_geometric.data import Data

def GNNDataLoader(filepath='facebook.npz'):
    """
    Custom dataloader for loading processed node data from file 'facebook.npz'
    We load the Facebook Large Page-Page Network dataset 
    Then convert it into a format suitable for use with PyTorch Geometric
    """
    # Load in data from file path as some object
    # From the original dataset, keys are: 'edges', 'features', 'target'
    data = np.load(filepath)
    
    # Extracting keys as tensors:
    edges = torch.tensor(data['edges'], dtype=torch.long) # connections between nodes
    features = torch.tensor(data['features'], dtype=torch.float) # 128-dim vects
    targets = torch.tensor(data['target'], dtype=torch.long) # targets for classification

    # Format edge indexes:
    # Format such that edges are in shape of [2, edge_num]
    # First row is source node, second row is target node
    edges = edges.t().contiguous()

    # Create a Data object using PyTorch Geometric
    data = Data(x=features, edge_index=edges, y=targets)

    return data






