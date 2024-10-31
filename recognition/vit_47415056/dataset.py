import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Constants for image processing
IMAGE_SIZE = 224  # Size to which images will be resized or cropped
BATCH_SIZE = 32  # Number of samples per batch for data loading

# Function to apply image transformations
def get_transforms(is_train=True):
    """
    Returns a set of transformations to apply to the images.

    Args: is_train (bool): If True, applies data augmentation for training. If False, applies standard transformations for testing/validation.

    Returns: transforms.Compose: Composed transformations to apply to images.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])