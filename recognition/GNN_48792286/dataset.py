# import necessary packages from library
import numpy as np
import torch
from torch_geometric.data import Data

''' Function to load and preprocess the data from a file
    Load the dataset from a .npz file
'''
def load_data(file_path):
    data = np.load(file_path)
    # Extract features, edges, and target labels from the loaded data

    features = data['features']
    edges = data['edges']
    targets = data['target']
    #Ensure it is 1D and if not 1D, extract it
    if targets.ndim == 1:
        labels = targets  
    else:
        labels = targets[:, 0]  

    # Convert features, edges, and labels into PyTorch tensors
    x = torch.tensor(features, dtype=torch.float)  
    y = torch.tensor(labels, dtype=torch.long)    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  

    # Create a PyTorch Geometric data object to represent the graph
    graph_data = Data(x=x, edge_index=edge_index, y=y)

    ''' Create a training mask to specify which nodes to use for training
        And 80% is for training
    '''
    num_nodes = graph_data.x.size(0) 
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)  
    train_mask[:int(num_nodes * 0.8)] = True 
    graph_data.train_mask = train_mask  
    # Return the prepared graph data object
    return graph_data  
