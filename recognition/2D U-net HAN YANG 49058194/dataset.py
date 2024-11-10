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
        ground_truth_dir (str, optional): Directory containing corresponding ground truth .npy files.
    
    Methods: 
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Loads and returns a sample and ground truth (if available) at the specified index.
    """ 
    def __init__(self, root_dir, ground_truth_dir=None):
        self.root_dir = root_dir
        self.ground_truth_dir = ground_truth_dir
        self.file_list = []
        
        # Collect paths for MRI images
        for root, dirs, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith('.npy'):
                    self.file_list.append(os.path.join(root, file_name))
        
        # If ground_truth_dir is specified, collect corresponding ground truth paths
        if self.ground_truth_dir:
            self.ground_truth_list = []
            for root, dirs, files in os.walk(ground_truth_dir):
                for file_name in files:
                    if file_name.endswith('.npy'):
                        self.ground_truth_list.append(os.path.join(root, file_name))
            
            # Ensure the number of images matches the ground truth files
            assert len(self.file_list) == len(self.ground_truth_list), \
                "Mismatch in the number of images and ground truth files."

    def __len__(self):
        # Return the number of MRI files in the dataset
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load MRI image
        img_path = self.file_list[idx]
        image = np.load(img_path)

        # Normalize image data to the range [0, 1]
        image = image / np.max(image)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # If ground truth is available, load and return it with the image
        if self.ground_truth_dir:
            ground_truth_path = self.ground_truth_list[idx]
            ground_truth = np.load(ground_truth_path)
            ground_truth = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            return image, ground_truth

        # Return only the image if no ground truth is provided
        return image
