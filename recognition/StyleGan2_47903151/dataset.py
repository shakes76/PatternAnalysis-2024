"""
Contains data loader for loading and preprocessing data
Created on 08/10/2024
"""

import torch
import torch.nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import *


def get_loader(log_resolution, batch_size, directory = "AD_NC/train"):
    """
    :param log_resolution: int, log2 of the desired image resolution, for example 8 for 256x256 images.
    :param batch_size: int, the batch size
    :param directory: string, the location of the image folder
    :return:
    gets images from the image folder, transform the images and set up a data loader with the given batch size.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((2 ** log_resolution, 2 ** log_resolution)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ),
        ]
    )
    dataset = ImageFolder(root=directory, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    print("Data loader loaded")
    return loader
