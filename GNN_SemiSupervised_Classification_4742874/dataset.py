"""
File: dataset.py
Description: Contains the data loader for loading and preprocessing 
    the Facebook Large Page-Page (FLPP) Network dataset.
Course: COMP3710 Pattern Recognition
Author: Liam Mulhern (S4742847)
Date: 26/10/2024
"""

import torch

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch_geometric.datasets import FacebookPagePage

FLPP_CATEGORIES = ['politicians', 'governmental organizations', 'television shows', 'companies']

class FLPP(Dataset):
    def __init__(self, root):
        super(FLPP, self).__init__()

        self.flpp_dataset: FacebookPagePage = FacebookPagePage(root=root)

    def __len__(self):
        return self.flpp_dataset.x.shape[0]

    def __getitem__(self, index):
        feature = self.flpp_dataset.x[index]
        label = self.flpp_dataset.y[index]

        return feature, label

def load_dataset(root: str, batch_size: int) -> tuple[FacebookPagePage, DataLoader, DataLoader, DataLoader]:
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
    train_size = int(0.7 * len(flpp_dataset))
    test_size = int(0.2 * len(flpp_dataset))
    validate_size = len(flpp_dataset) - train_size - test_size

    #train_sampler = SubsetRandomSampler(list(range(train_size)))
    #test_sampler = SubsetRandomSampler(list(range(train_size, test_size)))
    #validate_sampler = SubsetRandomSampler(list(range(test_size, validate_size)))

    train_dataset, test_dataset, validate_dataset = torch.utils.data.random_split(flpp_dataset, [0.8, 0.1, 0.1])

    # Load the Training dataset into memory
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        #sampler=train_sampler
    )

    # Load the Test dataset into memory
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        #sampler=test_sampler
    )

    # Load the Validate dataset into memory
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        #sampler=validate_sampler
    )

    return flpp_dataset.flpp_dataset, train_dataloader, test_dataloader, validate_dataloader
