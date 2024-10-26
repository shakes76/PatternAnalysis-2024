"""
File: dataset.py
Description: Contains the data loader for loading and preprocessing 
    the Facebook Large Page-Page (FLPP) Network dataset.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch

from torch.utils.data import Dataset, DataLoader
from torch_geometric.datasets import FacebookPagePage

FLPP_CATEGORIES = ['politicians', 'governmental organizations', 'television shows', 'companies']

def load_dataset(root: str, batch_size: int) -> tuple[FacebookPagePage, DataLoader, DataLoader]:
    """
        Load The Facebook Page-Page Network data set and separate 
        graph data into training and testing data loaders.

        Returns:
            Tuple (flpp_dataset, train_dataloader, test_dataloader)
    """

    # Load the FacebookPagePage dataset
    flpp_dataset: Dataset = FacebookPagePage(root=root)

    # Separate the dataset into training and testing sets
    train_size = int(0.8 * len(flpp_dataset))
    test_size = len(flpp_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(flpp_dataset, [train_size, test_size])

    # Load the Training dataset into memory
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    # Load the Test dataset into memory
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )

    return flpp_dataset, train_dataloader, test_dataloader


