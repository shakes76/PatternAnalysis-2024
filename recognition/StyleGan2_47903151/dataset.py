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
