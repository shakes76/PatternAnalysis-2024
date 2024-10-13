import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch


IMAGE_DIM = 224 

transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
    transforms.RandomResizedCrop(size=IMAGE_DIM, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Imagenet values, or adjust based on MRI dataset stats
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
])


def create_train_loader(root_dir, batch_size=32, val_split=0.2, seed=69):
    """
    Creates a DataLoader for the training set with data augmentation.

    Args:
        root_dir (str): Directory containing the training dataset.
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of data to use for validation.
        seed (int): Seed for reproducibility.

    Returns:
        DataLoader: A PyTorch DataLoader for the training set.
    """
    # Define transform for training data with augmentations
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.Grayscale(num_output_channels=1),
        #transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=10),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1),
        #transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(size=IMAGE_DIM, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
        #transforms.RandomErasing(p=0.3, scale=(0.02, 0.05), ratio=(0.3, 3.3)),
    ])
    
    dataset = datasets.ImageFolder(root=root_dir, transform=train_transform)
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, _ = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    return train_loader


# Function to create a validation DataLoader
def create_val_loader(root_dir, batch_size=32, val_split=0.2, seed=69):
    """
    Creates a DataLoader for the validation set without data augmentation.

    Args:
        root_dir (str): Directory containing the training dataset.
        batch_size (int): Number of samples per batch.
        val_split (float): Proportion of data to use for validation.
        seed (int): Seed for reproducibility.

    Returns:
        DataLoader: A PyTorch DataLoader for the validation set.
    """
    # Define transform for validation data (no augmentations)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    
    dataset = datasets.ImageFolder(root=root_dir, transform=val_transform)
    train_size = int(len(dataset) * (1 - val_split))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return val_loader


# Function to create a test DataLoader
def create_test_loader(root_dir, batch_size=32):
    """
    Creates a DataLoader for the test set without data augmentation.

    Args:
        root_dir (str): Directory containing the test dataset.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: A PyTorch DataLoader for the test set.
    """
    # Define transform for test data (no augmentations)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    
    dataset = datasets.ImageFolder(root=root_dir, transform=test_transform)

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return test_loader