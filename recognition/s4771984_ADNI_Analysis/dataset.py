import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from google.colab import drive

# Mounting the Google Drive
drive.mount('/content/drive',force_remount=True)

train_dir = '/content/drive/MyDrive/ADNI/AD_NC/train'
test_dir = '/content/drive/MyDrive/ADNI/AD_NC/test'

def get_data_loaders(train_dir, test_dir, batch_size = 64, val_split=0.2):
    # Now let's apply Data Augmentation techniques for the training and testing datasets
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(128),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    # Load the dataset from the google drive
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    
    #Finding out the data indices and shuffling them
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    split = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

