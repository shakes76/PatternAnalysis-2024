import numpy as np
from torch_geometric.data import Data
import torch

# 4 catoegories (politicians, governmental organizations, television shows and companies)
# Not directed 
# Nodes	22,470
# Edges	171,002

def load_data():
    # local directory
    data = np.load("/Users/jace/Desktop/Coding/Python/2024/COMP3710/Project/GNN/facebook.npz")

    # cluster directory
    # data = np.load(" /home/Student/s4722208/Project/facebook.npz")
    data['features']
    features = torch.tensor(data['features'])
    edges = torch.tensor(data['edges'])
    target = torch.tensor(data['target'])

    return Data(x = features, edge_index = edges, y = target )


