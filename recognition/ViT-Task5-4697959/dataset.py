# dataset.py

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
from sklearn.utils import class_weight
import numpy as np

def get_data_loaders(data_dir, batch_size=32, img_size=224, val_split=0.2, num_workers=4):
    """
    Creates training, validation, and test data loaders.

    Args:
        data_dir (str): Path to the ADNI dataset directory.
        batch_size (int): Batch size for data loaders.
        img_size (int): Size to resize the images.
        val_split (float): Fraction of training data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
        list: List of class names.
        torch.Tensor: Class weights for handling imbalance.
    """

    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Paths to dataset splits
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_val_transforms)

    # Split train_dataset into train and validation
    num_train = len(train_dataset)
    num_val = int(val_split * num_train)
    num_train = num_train - num_val
    train_subset, val_subset = random_split(train_dataset, [num_train, num_val])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class names
    class_names = train_dataset.classes

    # Compute class weights for handling class imbalance
    labels = [label for _, label in train_subset]
    class_weights = class_weight.compute_class_weight('balanced',
                                                     classes=np.unique(labels),
                                                     y=labels)
    # computed class weights are converted to a PyTorch
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, class_names, class_weights
