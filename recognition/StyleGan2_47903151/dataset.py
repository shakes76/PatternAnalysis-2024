"""
Contains data loader for loading and preprocessing data
Created on 08/10/2024
"""
from typing import Tuple, List, Dict, Union, Optional
from pathlib import Path
import torch
import torch.nn
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import *

def find_classes(directory: Union[str, Path],
                 desired_class_names: Optional[List]) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    if desired_class_names is None:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    else:
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes) if cls_name in desired_class_names}
    return classes, class_to_idx

class CustomImageFolder(ImageFolder):
    """
    Custom Image Folder allowing fetching selected classes while ignoring the rest.
    """
    def __init__(self, root, transform, desired_classes: List):
        self.desired_classes = desired_classes
        super().__init__(root, transform)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.
        Overwritten method allowing us to choose to only fetch classes we desired.
        """
        return find_classes(directory=directory, desired_class_names=self.desired_classes)


def get_loader(log_resolution, batch_size, directory="AD_NC/train", classes=""):
    """
    :param log_resolution: int, log2 of the desired image resolution, for example 8 for 256x256 images.
    :param batch_size: int, the batch size
    :param directory: string, the location of the image folder
    :param classes: string (instead of a list) because there is only 3 cases to be considered. Empty string returns
        loader with both classes, AD returns loader with only AD class, NC returns only NC class.
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
    if classes == "AD":
        dataset = CustomImageFolder(root=directory, transform=transform, desired_classes=["AD"])
        print("Training only on AD dataset")
    elif classes == "NC":
        dataset = CustomImageFolder(root=directory, transform=transform, desired_classes=["NC"])
        print("Training only on NC dataset")
    else:
        dataset = CustomImageFolder(root=directory, transform=transform, desired_classes=None)
        print("Training of all classes")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    print("Data loader loaded")
    return loader
