"""
File: dataset.py
Description: Contains the data loader for loading and preprocessing 
    the Facebook Large Page-Page (FLPP) Network dataset.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

FLPP_CATEGORIES = ['politicians', 'governmental organizations', 'television shows', 'companies']

class FLPP(Dataset):
    def __init__(self, root):
        super(FLPP, self).__init__()

        data = np.load(f"{root}/facebook.npz")

        # Load the numpy arrays into tensors
        self.features = torch.tensor(data['features'], dtype=torch.float32)
        self.edges = torch.tensor(data['edges'], dtype=torch.int64)
        self.edges = self.edges.t()
        self.target = torch.tensor(data['target'], dtype=torch.int64)
        self.num_nodes = self.features.size(0)

    def __len__(self):
        return self.num_nodes

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.target[index]

        return feature, label

def load_dataset(root: str, batch_size: int) -> tuple[FLPP, DataLoader, DataLoader, DataLoader]:
    """
        Load The Facebook Page-Page Network data set and separate 
        graph data into training and testing data loaders.

        Train Size = 70%
        Test Size = 20%
        Validate Size = 10%

        Parameters:
            root: The root directory of the raw dataset
            batch_size: The size of the dataset subdivisions

        Returns:
            Tuple (flpp_dataset, train_dataloader, test_dataloader)
    """

    # Load the FacebookPagePage dataset
    flpp_dataset: FLPP = FLPP(root=root)

    # Separate the dataset into training and testing sets
    train_dataset, test_dataset, validate_dataset = torch.utils.data.random_split(flpp_dataset, [0.8, 0.1, 0.1])

    # Load the Training dataset into memory
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Load the Test dataset into memory
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # Load the Validate dataset into memory
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return flpp_dataset, train_dataloader, test_dataloader, validate_dataloader
