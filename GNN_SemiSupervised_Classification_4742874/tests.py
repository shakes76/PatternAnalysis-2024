"""
File: tests.py
Description: Unit tests for the models and data set implementation.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import unittest

import modules
import dataset
import utils

DATASET_DIR = "./dataset/"

class TestDataSet(unittest.TestCase):
    def test_load_dataset(self):
        """
            Tests loading .npz file from FLPPDataset
        """
        flpp_dataset, training_data, testing_data = dataset.load_dataset(DATASET_DIR, 200)

        print('Dataset properties')
        print('==============================================================')
        print(f'Dataset: {flpp_dataset}') #This prints the name of the dataset
        print(f'Number of graphs in the dataset: {len(flpp_dataset)}')
        print(f'Number of features: {flpp_dataset.num_features}') #Number of features each node in the dataset has
        print(f'Number of classes: {flpp_dataset.num_classes}') #Number of classes that a node can be classified into
        print(f'Number of nodes: {flpp_dataset.x.shape[0]}')

        assert len(flpp_dataset) == 1
        assert flpp_dataset.num_features == 128
        assert flpp_dataset.num_classes == 4
        assert flpp_dataset.x.shape[0] == 22470

class TestUtils(unittest.TestCase):
    def test_utils_display_graph(self):
        """
            Tests creating teh spring layout of the FLPP graph
            visualising category connections.
        """
        flpp_dataset, training_data, testing_data = dataset.load_dataset(DATASET_DIR, 200)

        utils.display_flpp_network(flpp_dataset)

class TestModules(unittest.TestCase):
    def test_modules_create(self):
        """
        """
