import numpy as np
import torch
from torch_geometric.data import Data

def load_dataset(file_path='facebook.npz'):
    data = np.load(file_path)
    features = torch.tensor(data['features'], dtype=torch.float)
    edges = torch.tensor(data['edges'], dtype=torch.long)
    labels = torch.tensor(data['target'], dtype=torch.long)
    return Data(x=features, edge_index=edges, y=labels)