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

DATASET_DIR = "./dataset/raw"

class TestDataSet(unittest.TestCase):
    def test_load_dataset(self):
        """
            Tests loading .npz file from FLPPDataset
        """
        flpp_dataset, training_data, testing_data, validate_data = dataset.load_dataset(DATASET_DIR, 200)

        assert len(flpp_dataset) == 22470

        print(len(training_data))
        print(len(testing_data))
        print(len(validate_data))

class TestUtils(unittest.TestCase):
    def test_utils_display_raw_tsne(self):
        """
            Tests creating the TSNE display for the raw dataset.
        """
        flpp_dataset, training_data, testing_data, validate_data = dataset.load_dataset(DATASET_DIR, 200)

        utils.display_raw_tsne(flpp_dataset)

class TestModules(unittest.TestCase):
    def test_modules_create(self):
        """
        """
