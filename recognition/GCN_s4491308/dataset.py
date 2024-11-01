"""
Author: Ananya Dubey 
Student No : 44913083 
This script contains code to load, process and split the dataset from the facebook.npz file. 
"""

import torch
from torch_geometric.data import Data
import numpy as np

def load_data(data_file_path):
    # Load data from the npz file 
    data = np.load(data_file_path)
    #check for arrays stored in npz file 
    #print ("Keys:", list(data.keys()))
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
    data = Data(x=x, edge_index=edges, y=y)
    #print("graph data object", data)

    return data 


def index_to_mask(index, size):
    """
    Function to get a boolen mask of given size for the given indices 
    Reference: Inspired and adapted from code at https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/mask.py#L41
    """
    mask = index.new_zeros(size, dtype = torch.bool)
    mask[index] = True 
    return mask 

def perform_split(data, train_ratio, validation_ratio, test_ratio):
    """
    Function to perform the train, test and validation splits. 
    Reference: Inspired and adapted from code at https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/mask.py#L41
    """
    #total nodes in the dataset
    total_nodes = data.num_nodes
    #randomly shuffle the nodes 
    indices = torch.randperm(total_nodes)
    # sizes for the train, test and validation sets 
    train_size = int(total_nodes * train_ratio)
    validation_size = int(total_nodes * validation_ratio)
    test_size = total_nodes - train_size - validation_size
    # indices for the test, train and validation data
    train_indices = indices[ :train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size: ]
    # getting the masks for each set 
    data.train_mask = index_to_mask(train_indices,total_nodes)
    data.validation_mask = index_to_mask(validation_indices, total_nodes)
    data.test_mask = index_to_mask(test_indices, total_nodes)
    return data.train_mask, data.validation_mask, data.test_mask

# Test for the dataloader 
# file_path = '/content/facebook.npz'  # Replace with the path to your .npz file
# data = load_data(file_path)
# train_mask, validation_mask, test_mask = perform_split(data, 0.80, 0.10, 0.10)
# # checking splits 
# total_nodes = data.num_nodes
# print(f"Total nodes: {total_nodes}")
# print(f"Train nodes (expected {int(0.80 * total_nodes)}): {train_mask.sum().item()}")
# print(f"Validation nodes (expected {int(0.10 * total_nodes)}): {validation_mask.sum().item()}")
# print(f"Test nodes (expected {int(0.10 * total_nodes)}): {test_mask.sum().item()}")

# check for overlap
# train_and_val_overlap = (train_mask & validation_mask).sum().item()
# train_and_test_overlap = (train_mask & test_mask).sum().item()
# val_and_test_overlap = (validation_mask & test_mask).sum().item()

# print(f"Train and validation overlap: {train_and_val_overlap}")
# print(f"Train and test overlap: {train_and_test_overlap}")
# print(f"Validation and test overlap: {val_and_test_overlap}")



