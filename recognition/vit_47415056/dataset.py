import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Constants for image processing
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Function to apply image transformations
def get_transforms(is_train=True):
    """
    Returns a set of transformations to apply to the images.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1415] * 3, std=[0.2420] * 3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1415] * 3, std=[0.2420] * 3),
        ])

# Function to create training and validation loaders
def get_train_val_loaders(data_dir):
    transform = get_transforms(is_train=True)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return train_loader, val_loader

# Function to create a test loader
def get_test_loader(data_dir):
    transform = get_transforms(is_train=False)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)