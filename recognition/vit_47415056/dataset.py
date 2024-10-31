import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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

# Function to create training and validation loaders
def get_train_val_loaders(data_dir):
    transform = get_transforms(is_train=True)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Split dataset: 90% for training, 10% for validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
    return train_loader, val_loader