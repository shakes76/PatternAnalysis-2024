"""
Implemented to handle the loading and preprocessing of the data in the
ISIC 2020 Kaggle Challenge data set.

Made by Joshua Deadman
"""

import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import random
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

from utils import split_data
from config import BATCH_SIZE, DATAPATH, WORKERS

# Code is inspired by that found in:
#   https://github.com/andreasveit/triplet-network-pytorch/blob/master/triplet_mnist_loader.py
class ISICKaggleChallengeSet(Dataset):
    """ Object that stores all data used in training, testing and validation. """
    def __init__(self, root, image_set, transforms=None) -> None:
        """ Initialises the dataset.

        Arguments:
            root (str): The absolute path to the image dataset.
            image_set (list): A list from the utils.split_data() method.
            transforms (obj): A v2 instance of transformations.
                    The transforms must include a .ToImage().
        """
        self._root = root
        self._image_set = image_set
        self._transforms = transforms
   
    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """ Returns a random triplet.

        Arguments:
            index (int): The index into the dataset.
        Returns:
            A tuple with the anchor, positive, negative and label of the the anchor.
        """
        anchor = self._image_set[index]
        positive = random.choice(self._image_set)
        while positive["target"] != anchor["target"]:
            positive = random.choice(self._image_set)
        negative = random.choice(self._image_set)
        while negative["target"] == anchor["target"]:
            negative = random.choice(self._image_set)

        anchor_img = Image.open(os.path.join(self._root, anchor["isic_id"] + ".jpg")).convert("RGB")
        positive_img = Image.open(os.path.join(self._root, positive["isic_id"] + ".jpg")).convert("RGB")
        negative_img = Image.open(os.path.join(self._root, negative["isic_id"] + ".jpg")).convert("RGB")

        if self._transforms is not None:
            anchor_img = self._transforms(anchor_img)
            positive_img = self._transforms(positive_img)
            negative_img = self._transforms(negative_img)
        else:
            to_tensor = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ])
            anchor_img = to_tensor(anchor_img)
            positive_img = to_tensor(positive_img)
            negative_img = to_tensor(negative_img)

        return anchor_img, positive_img, negative_img, anchor["target"]

    def __len__(self) -> int:
        """ Returns the size of the dataset. """
        return len(self._image_set)

    def show_images(self) -> None:
        """ Plots 9 random images applying any transforms present. """
        images = []
        indices = np.random.randint(0, len(self._image_set), size=3)
        for i in indices:
            images.append(self[i])
        figure = 1
        for y, tup in enumerate(images):
            for x, image in enumerate(tup):
                if x  < 3: # Don't try to plot label
                    plt.subplot(3,3,figure)
                    plt.imshow(image.permute(1,2,0).numpy())
                    if x == 0 or x == 1:
                        label = "Benign" if images[y][-1] == 0 else "Malignant"
                    else:
                        label = "Benign" if images[y][-1] == 1 else "Malignant"
                    plt.title(label)
                    figure += 1
                    plt.axis("off")
        plt.show()

def get_dataloaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    """ Performs the splitting, and loading of the training, testing and validation sets. 

    Returns:
        A tuple of the DataLoader objects for the training, testing and validations sets respectively.  
    """
    # Form the disjoin sets of data
    train, test, val = split_data(os.path.join(DATAPATH, "train-metadata.csv"))

    transforms = v2.Compose([
        v2.RandomRotation(degrees=(0, 10)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Form datsets and load them
    train_set = ISICKaggleChallengeSet(os.path.join(DATAPATH, "train-image/image/"), train, transforms=transforms)
    test_set = ISICKaggleChallengeSet(os.path.join(DATAPATH, "train-image/image/"), test, transforms=transforms)
    val_set = ISICKaggleChallengeSet(os.path.join(DATAPATH, "train-image/image/"), val, transforms=transforms)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=WORKERS)

    return train_loader, test_loader, val_loader
