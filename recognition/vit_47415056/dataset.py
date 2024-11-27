import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Constants for image processing
IMAGE_SIZE = 224
BATCH_SIZE = 32

def get_transforms(is_train=True):
    """
    Returns transformations for training or testing images.
    
    Args:
        is_train (bool): True for training transforms, False for testing.
        
    Returns:
        torchvision.transforms.Compose: Composed transformations.
    """
    if is_train:
        # Data augmentation for training
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1415] * 3, std=[0.2420] * 3),
        ])
    else:
        # Basic transforms for testing
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1415] * 3, std=[0.2420] * 3),
        ])

def get_train_val_loaders(data_dir):
    """
    Creates DataLoaders for training and validation sets.
    
    Args:
        data_dir (str): Directory with image data.
        
    Returns:
        tuple: (train_loader, val_loader) for training and validation.
    """
    transform = get_transforms(is_train=True)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split dataset into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return train_loader, val_loader

def get_test_loader(data_dir):
    """
    Creates a DataLoader for the test set.
    
    Args:
        data_dir (str): Directory with test image data.
        
    Returns:
        DataLoader: DataLoader for the test set.
    """
    transform = get_transforms(is_train=False)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # DataLoader for test dataset
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)