# -----------------------------------------------------------
# Project: Mixed Graph Neural Networks for Node Classification
# Filename: dataset.py
# Author: Tsung-Tse Tu
# Student ID: s4780187
# Date: October 2024 (Last edited 10/21/2024)
# Description: This script contains the function for loading
#              the Facebook Large Page-Page Network Dataset 
#              from a .npz file.
# -----------------------------------------------------------


import numpy as np

def load_facebook_data():
    dataset_path = "./data/facebook.npz"
    data = np.load(dataset_path)
    return data


