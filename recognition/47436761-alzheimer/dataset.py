import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

# Define the paths
train_dir = 'dataset/AD_NC/train'
test_dir = 'dataset/AD_NC/test'
batch_size = 32

IMAGE_DIM = 224 
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_DIM // PATCH_SIZE) ** 2
D_MODEL = (PATCH_SIZE ** 2) * 3

transform = transforms.Compose([
    transforms.Resize((IMAGE_DIM, IMAGE_DIM)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMAGE_DIM, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
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
    print(f"Using the full dataset from {root_dir} with {len(dataset)} samples")

    # Create a local generator for reproducibility
    generator = torch.Generator().manual_seed(seed)

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


if __name__ == "__main__":
    # Specify the subset size (e.g., 100 samples)
    train_subset_size = 100  # Set to None to use the full training set
    test_subset_size = 50    # Set to None to use the full test set

    # Create DataLoaders for training and testing
    train_loader = create_data_loader(train_dir, batch_size=batch_size, train=True, subset_size=train_subset_size)
    test_loader = create_data_loader(test_dir, batch_size=batch_size, train=False, subset_size=test_subset_size)

    print(f"Class-to-index mapping: {train_loader.dataset.dataset.class_to_idx}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of testing samples: {len(test_loader.dataset)}")
