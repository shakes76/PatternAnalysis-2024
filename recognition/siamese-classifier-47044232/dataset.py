"""
Implemented to handle the loading and preprocessing of the data in the
ISIC 2020 Kaggle Challenge data set.

Made by Joshua Deadman
"""

import matplotlib as plt
import pandas as pd
import random
import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

from config import TRAINING, TESTING, VALIDATION

# Code is inspired by that found in:
#   https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_mnist_loader.py
class ISIC_Kaggle_Challenge_Set(Dataset):
    """ Object that stores all data used in training, testing and validation. """
    def __init__(self, root, label_path, images, transforms=None) -> None:
        """ Initialises the dataset.

        Arguments:
            root (str): The absolute path to the image dataset.
            label_path (str): The absolute path to the label's .csv file.
            images (list): A list of images to be used in the data set.
            transforms (obj): A v2 instance of transformations.
        """
        self._root = root
        self._df = pd.read_csv(label_path)
        self._images = images
        random.shuffle(self._images)
        # TODO confirm set_type selections work.
        if len(self._images) == len(self._df.index) * TRAINING:
            self._set_type = TRAINING
        elif len(self._images) == len(self._df.index) * TESTING:
            self._set_type = TESTING
        else:
            self._set_type = VALIDATION
        self._transforms = transforms
   
    # TODO implement this to return a triplet with augmentations applied.
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """ Returns a random triplet.

        Arguments:
            index (int): The index into the dataset.
        Returns:
            A tuple with the anchor, positive, negative and label of the the anchor.
        """
        pass

    def __len__(self) -> int:
        """ Returns the size of the dataset. """
        return len(self._images)

    # TODO implement this method.
    def show_images(self) -> None:
        """ Shows a few images with any transforms applied. """
        pass
