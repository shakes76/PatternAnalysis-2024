import torch
from torch_geometric.data import Data
import numpy as np

def load_data(data_file_path):
    # Load data from the npz file 
    data = np.load(data_file_path)
    #check for arrays stored in npz file 
    print ("Keys:", list(data.keys()))
    #features 
    features = data['features']
    #edges
    edges = data['edges']
    #target 
    target = data['target']
    #check shape 
    #print(f"Features shape:{features.shape}, Edges shape:{edges.shape}, Target Shape:{target.shape}")
    #load the arrays to tensors 
    x = torch.tensor(features, dtype=torch.float)
    edges = torch.tensor(edges, dtype=torch.long).T.contiguous()
    y = torch.tensor(target, dtype=torch.long)
    #creating data object for PyTorch geometric 
    data = Data(x=features, edge_index=edges, y=target)
    #print("graph data object", data)

    return data 


