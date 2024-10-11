import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch


IMAGE_DIM = 224 

transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]) # Imagenet values
])


def create_data_loader(root_dir, batch_size=32, train=True, val=False, val_split=0.2, seed=69):
    """
    Creates a DataLoader for the given dataset directory, with an optional split for validation.

    Args:
        root_dir (str): Directory containing the dataset (train or test).
        batch_size (int): Number of samples per batch.
        train (bool): Whether this is a training loader (shuffles data if True).
        val (bool): Whether to split the data into training and validation.
        val_split (float): Proportion of data to use for validation if splitting.

    Returns:
        DataLoader or tuple of DataLoader: A PyTorch DataLoader for the dataset, or a tuple of
        (train_loader, val_loader) if `val=True`.
    """
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    

    # Create a local generator for reproducibility
    generator = torch.Generator().manual_seed(seed)


    fraction = 0.7 # 0.3
    subset_size = int(len(dataset) * fraction)

    # Generate a random permutation of indices
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size]
    dataset = Subset(dataset, indices)
    print(f"Using the full dataset from {root_dir} with {len(dataset)} samples")

    # If val is True, split the dataset into training and validation sets
    if val:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

        # Create DataLoaders for training and validation sets
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Do not shuffle validation data
            num_workers=4,
            pin_memory=True
        )
        return train_loader, val_loader

    # If val is False, create a standard DataLoader (for training or testing)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,  # Shuffle if training, do not shuffle if testing
        num_workers=4,
        pin_memory=True
    )
    return loader

