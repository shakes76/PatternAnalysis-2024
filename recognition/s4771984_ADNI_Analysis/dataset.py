import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

def get_data_loaders(train_dir, test_dir, batch_size=64):
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(128),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load train dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    train_dataset.class_to_idx = {'NC': 0, 'AD': 1}

    # Get the indices for each class
    targets = np.array([sample[1] for sample in train_dataset.samples])
    nc_indices = np.where(targets == train_dataset.class_to_idx['NC'])[0]
    ad_indices = np.where(targets == train_dataset.class_to_idx['AD'])[0]

    # Shuffle and split indices for AD and NC class (70% train, 30% validation)
    np.random.shuffle(nc_indices)
    np.random.shuffle(ad_indices)

    train_size_nc = int(0.7 * len(nc_indices))
    train_size_ad = int(0.7 * len(ad_indices))

    nc_train_indices = nc_indices[:train_size_nc]
    nc_val_indices = nc_indices[train_size_nc:]

    ad_train_indices = ad_indices[:train_size_ad]
    ad_val_indices = ad_indices[train_size_ad:]

    # Combine indices
    train_indices = np.concatenate([nc_train_indices, ad_train_indices])
    val_indices = np.concatenate([nc_val_indices, ad_val_indices])

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoader for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=12)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=12)

    # Load test data
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
    test_dataset.class_to_idx = {'NC': 0, 'AD': 1}
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    return train_loader, val_loader, test_loader