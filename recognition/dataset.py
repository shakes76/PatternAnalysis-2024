import numpy as np
from torch_geometric.data import Data
import torch

# 4 catoegories (politicians, governmental organizations, television shows and companies)
# Not directed 
# Nodes	22,470
# Edges	171,002x    

def load_data():
    # local directory
    data = np.load("/Users/jace/Desktop/Coding/Python/2024/COMP3710/Project/GNN/facebook.npz")

    # cluster directory
    # data = np.load(" /home/Student/s4722208/Project/facebook.npz")
    features = torch.tensor(data['features'], dtype=torch.float32)
    edges = torch.tensor(data['edges'], dtype=torch.int64)
    edges = edges.t()
    target = torch.tensor(data['target'], dtype=torch.int64)


    number_of_nodes = features.size(0)

    train_size = int(number_of_nodes * 0.7)
    val_size = int(number_of_nodes * 0.2)    
    test_size = number_of_nodes - train_size - val_size

    train_mask = torch.zeros(number_of_nodes, dtype=torch.bool)
    val_mask = torch.zeros(number_of_nodes, dtype=torch.bool)
    test_mask = torch.zeros(number_of_nodes, dtype=torch.bool)

    node_shuffled = np.arange(number_of_nodes)
    np.random.shuffle(node_shuffled)

    train_node = node_shuffled[:train_size]
    validation_node = node_shuffled[train_size:train_size+val_size]
    test_node = node_shuffled[-test_size:]

    train_mask[train_node] = 1
    val_mask[validation_node] = 1
    test_mask[test_node] = 1


    return (Data(x = features, edge_index = edges, y = target), [train_mask, test_mask, val_mask])
