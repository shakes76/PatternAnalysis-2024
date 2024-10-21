""" The functions in this file are loading and preprocessing
    the Facebook Page-Page Large Network dataset
"""

import numpy as np
import scipy.sparse as sp
import torch
FILE_PATH = "./facebook.npz"


""" Creates a class instance that will load the data on construction
    and prepare it for training
"""
class FacebookPagePageLargeNetwork():

    def __init__(self,
                 file_path: str,
                 test_ratio: float,
                 validation_ratio: float
                 ) -> None:

        self.data = np.load(file_path)
        self.target = self.data["target"]
        self.edges = self.data["edges"]
        self.features = self.data["features"]

        num_nodes = len(self.target)
        size_test = int(num_nodes * test_ratio)
        size_validation = int(num_nodes * validation_ratio)
        size_train = int(num_nodes * (1 - test_ratio - validation_ratio))
        
        np.random.seed(31)
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        train_idx, val_idx, test_idx = indices[:size_train], indices[size_train:size_train+size_validation], indices[-size_test:]


        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = 1

        y = torch.tensor(self.target, dtype=torch.int64)
        y_true = y[train_mask]
        print(train_mask)
        # print(len(train_mask[train_idx]))

        # Creating an adjacency matrix from the edges information
        self.adjacency_matrix = sp.coo_matrix((np.ones(len(self.edges)), (self.edges[:, 0], self.edges[:, 1])),
                        shape=(num_nodes, num_nodes))
        self.adjacency_matrix = torch.tensor(self.adjacency_matrix.todense())



        print(f"test len {size_test}\nval len {size_validation}\ntrain len {size_train}")

    def _load_data(filepath: str):

        data = np.load(filepath)

        target = data["target"]
        edges = data["edges"]
        features = data["features"]

        print(f"target len: {len(target)}\nedges len: {len(edges)}\nfeatures len: {len(features)}")



        print(edges)
        return data

if __name__ == "__main__":
    data = FacebookPagePageLargeNetwork(FILE_PATH, 0.1, 0.1)

    # print(f"Time taken to load the data: {(end_time-start_time)} seconds")

    # print(len(data["edges"]))
   