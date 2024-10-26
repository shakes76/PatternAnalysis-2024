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

from typing import Any
from torch.utils.data import Dataset

FLPP_CATEGORIES = ['politicians', 'governmental organizations', 'television shows', 'companies']

class FLPPDataset(Dataset):
    def __init__(self, image_dir: str) -> None:
        """
            Initialise the FLPP dataset from file 
            and apply normalisation and transformation.

            Parameters:
                image_dir: The directory to load the FLPP .npz dataset
        """
        # Open dataset
        with np.load(image_dir) as data:
            self.edges = torch.tensor(data['edges'])
            self.features = torch.tensor(data['features'])
            self.labels = torch.tensor(data['target'])

    def __len__(self) -> int:
        """
            Get thh lenght of the load FLPP dataset
        """
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """
            Get an element at the given index from the FLPP 
            dataset.

            Parameters:
                index: The index of the dataset to get the element

            Returns:
                A feature and label at the given index.
        """
        return self.features[index], self.labels[index]
