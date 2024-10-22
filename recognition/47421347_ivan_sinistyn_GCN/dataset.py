""" The functions in this file are loading and preprocessing
    the Facebook Page-Page Large Network dataset
"""

import numpy as np
import scipy.sparse as sp
import torch


""" Creates a class instance that will load the data on construction
    and prepare it for training
"""
class FacebookPagePageLargeNetwork():

    def __init__(self,
                 file_path: str,
                 test_ratio: float,
                 validation_ratio: float,
                 seed: int
                 ) -> None:

        self.data = np.load(file_path)
        self.target = self.data["target"]
        self.edges = self.data["edges"]
        self.features = self.data["features"]

        num_nodes = len(self.target)
        size_test = int(num_nodes * test_ratio)
        size_validation = int(num_nodes * validation_ratio)
        size_train = int(num_nodes * (1 - test_ratio - validation_ratio))

        # Shuffle the data
        np.random.seed(seed)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        # Create arrays of indices for train, validation and test
        train_indices, val_indices, test_indices = indices[:size_train], indices[size_train:size_train+size_validation], indices[-size_test:]

        # Create mask for filtering out only the test/train/validation nodes from dataset
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.train_mask[train_indices] = 1

        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask[test_indices] = 1

        self.validation_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.validation_mask[val_indices] = 1


        self.y = torch.tensor(self.target, dtype=torch.int64)
        self.x = torch.tensor(self.features, dtype=torch.float32)

        # Separate the data into subsets for train, test and validation
        self.x_train, self.y_train = self.x[self.train_mask], self.y[self.train_mask]
        self.x_test, self.y_test = self.x[self.test_mask], self.y[self.test_mask]
        self.x_val, self.y_val = self.x[self.validation_mask], self.y[self.validation_mask]

        # Creating an adjacency matrix from the edges information
        self.adjacency_matrix = sp.coo_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
        self.adjacency_matrix = torch.tensor(self.adjacency_matrix.todense())

    # Convet the tensors to CUDA (or any other device)
    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.adjacency_matrix = self.adjacency_matrix.to(device)

        self.train_mask = self.train_mask.to(device)
        self.test_mask = self.test_mask.to(device)
        self.validation_mask = self.validation_mask.to(device)

        self.x_train, self.y_train = self.x_train.to(device), self.y_train.to(device)
        self.x_test, self.y_test = self.x_test.to(device), self.y_test.to(device)
        self.x_val, self.y_val = self.x_val.to(device), self.y_val.to(device)

        return self
