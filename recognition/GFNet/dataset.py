import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from modules import data_transforms

def get_datasets(data_dir, val_split=0.15):
    """
    Loads and splits the datasets.

    Args:
        data_dir (str): Root directory containing 'Train' and 'Test' subdirectories.
        val_split (float): Proportion of training data to be used for validation.

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Paths to Train and Test directories
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    print(f"Checking existence of Train directory: {train_dir}")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found at: {train_dir}")
    print(f"Train directory found: {train_dir}")

    print(f"Checking existence of Test directory: {test_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found at: {test_dir}")
    print(f"Test directory found: {test_dir}")

    # Load the full training dataset
    print("Loading training dataset...")
    train_dataset_full = ImageFolder(root=train_dir, transform=data_transforms['train'])
    print(f"Total training samples: {len(train_dataset_full)}")

    # Calculate sizes for training and validation splits
    total_train_size = len(train_dataset_full)
    val_size = int(val_split * total_train_size)
    train_size = total_train_size - val_size

    print(f"Splitting training data into {train_size} training samples and {val_size} validation samples...")
    # Split the training dataset into training and validation
    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Training samples after split: {len(train_dataset)}")
    print(f"Validation samples after split: {len(val_dataset)}")

    # Update transforms for the validation set
    print("Updating transforms for validation dataset...")
    val_dataset.dataset.transform = data_transforms['val']

    # Load the test dataset
    print("Loading test dataset...")
    test_dataset = ImageFolder(root=test_dir, transform=data_transforms['test'])
    print(f"Total test samples: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset

def get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=0):
    """
    Creates DataLoader objects for training, validation, and testing.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Test dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    print("Training DataLoader created.")

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    print("Validation DataLoader created.")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    print("Test DataLoader created.")

    return train_loader, val_loader, test_loader
