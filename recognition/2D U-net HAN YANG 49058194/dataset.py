"""
dataset.py
----------
Custom dataset class for loading MRI slices and segmentation masks.

Input:
    - Root directory containing the dataset in .npy format.

Output:
    - Torch Dataset objects that can be used for training/testing with DataLoader.

Usage:
    Import this module to create a Dataset instance for training and evaluation.

Author: Han Yang
Date: 25/09/2024
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Define dataset, loading prostate MRI image data
class ProstateMRIDataset(Dataset):
   """ 
    Custom Dataset for loading and processing Prostate MRI data. 
    Args:
        root_dir (str): Directory containing .npy files of MRI slices. 
    Methods: 
        __len__(): Returns the number of samples in the dataset. 
        __getitem__(idx): Loads and returns a sample at the specified index. 
   """ 
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = []

        # Traverse every root directory
        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith('.npy'):
                    # Save files that end with .npy
                    self.file_list.append(os.path.join(root, file_name))

    
    # Return the number of MRI files in the dataset
    def __len__(self):
        return len(self.file_list)

    
    # Load and process MRI images with specified index
    def __getitem__(self, idx):
        img_path = self.file_list[idx] # Retrieve the path to the MRI file
        image = np.load(img_path) # Load MRI image data.
        
        # Normalize image data to the range of [0,1]
        image = image / np.max(image) 
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        return image
