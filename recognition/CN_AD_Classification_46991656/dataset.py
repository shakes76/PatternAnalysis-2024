# Contains the dataloader for loading and preprocessing data

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=32):
    # Data augmentations for the training set
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(10),  # Random rotation within 10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ])

    # No augmentations for the test set, only resize and normalization
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert images to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ])

    # Create datasets for training and testing
    train_dataset = datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root='/home/groups/comp3710/ADNI/AD_NC/test', transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


