"""
File: dataset.py
Description: Contains the data loader for loading and preprocessing 
    the Facebook Large Page-Page (FLPP) Network dataset.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data

FLPP_CATEGORIES = ['politicians', 'governmental organizations', 'television shows', 'companies']

class FLPP(Dataset):
    def __init__(self, root):
        super(FLPP, self).__init__()

        # Load numpy array from file
        data = np.load(f"{root}/facebook.npz")

        # Load the numpy arrays into tensors
        self.features = torch.tensor(data['features'], dtype=torch.float32)
        self.edges = torch.tensor(data['edges'], dtype=torch.int64)
        self.edges = self.edges.t()
        self.target = torch.tensor(data['target'], dtype=torch.int64)
        self.num_nodes = self.features.size(0)

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.target[index]

        return feature, label

def load_dataset(root: str):
    """
        Load The Facebook Page-Page Network data set and separate 
        graph data into training and testing data loaders.

        Train Size = 70%
        Test Size = 20%
        Validate Size = 10%

        Parameters:
            root: The root directory of the raw dataset
            batch_size: The size of the dataset subdivisions

        Returns:
            Tuple (flpp_dataset, train_dataloader, test_dataloader)
    """

    # Load the FacebookPagePage dataset
    flpp_dataset: FLPP = FLPP(root=root)

    train_size: int = int(0.8 * flpp_dataset.num_nodes)
    test_size: int = int(0.1 * flpp_dataset.num_nodes)
    validate_size: int = int(0.1 * flpp_dataset.num_nodes)

    # Separate the dataset into training and testing sets
    train_mask = torch.zeros(flpp_dataset.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(flpp_dataset.num_nodes, dtype=torch.bool)
    validate_mask = torch.zeros(flpp_dataset.num_nodes, dtype=torch.bool)

    # Randomly sample the dataset for the train, test, and validate samples
    node_shuffled = np.arange(flpp_dataset.num_nodes)
    np.random.shuffle(node_shuffled)

    train_nodes = node_shuffled[:train_size]
    validate_nodes = node_shuffled[train_size:train_size + validate_size]
    test_nodes = node_shuffled[train_size + validate_size:]

    train_mask[train_nodes] = True
    test_mask[test_nodes] = True
    validate_mask[validate_nodes] = True

    # Load the Training dataset into memory
    flpp_data = Data(x = flpp_dataset.features, edge_index = flpp_dataset.edges, y = flpp_dataset.target)

    return flpp_dataset, flpp_data, train_mask, test_mask, validate_mask
