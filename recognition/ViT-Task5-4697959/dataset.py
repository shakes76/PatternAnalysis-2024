# dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from sklearn.utils import class_weight
import numpy as np

def get_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, num_workers=4):
    """
    Creates training, validation, and test data loaders.

    Args:
        data_dir (str): Path to the ADNI dataset directory.
        batch_size (int): Batch size for data loaders.
        img_size (int): Size to resize the images.
        val_split (float): Fraction of training data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
        list: List of class names.
        torch.Tensor: Class weights for handling imbalance.
    """

    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

