"""
File: tests.py
Description: Unit tests for the models and data set implementation.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import unittest

from torch_geometric.datasets import FacebookPagePage

DATASET_DIR = "./dataset/"
FLPP_DATASET = "raw/facebook.npz"

class TestDataSet(unittest.TestCase):
    def test_load_dataset(self):
        """
            Tests loading .npz file from FLPPDataset
        """
        dataset = FacebookPagePage(DATASET_DIR)

        print('Dataset properties')
        print('==============================================================')
        print(f'Dataset: {dataset}') #This prints the name of the dataset
        print(f'Number of graphs in the dataset: {len(dataset)}')
        print(f'Number of features: {dataset.num_features}') #Number of features each node in the dataset has
        print(f'Number of classes: {dataset.num_classes}') #Number of classes that a node can be classified into
        print(f'Number of nodes: {dataset.x.shape[0]}')

        assert len(dataset) == 1
        assert dataset.num_features == 128
        assert dataset.num_classes == 4
        assert dataset.x.shape[0] == 22470

