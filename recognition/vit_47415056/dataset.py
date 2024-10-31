import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Constants for image processing
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Function to apply image transformations
def get_transforms(is_train=True):
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

# Updated data loader logic to use `get_transforms`
def get_dataloaders(data_dir):
    transform = get_transforms(is_train=True)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Create DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloader