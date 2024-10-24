import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from google.colab import drive

# Mounting the Google Drive
drive.mount('/content/drive',force_remount=True)

train_dir = '/content/drive/MyDrive/ADNI/AD_NC/train'
test_dir = '/content/drive/MyDrive/ADNI/AD_NC/test'

def get_data_loaders(train_dir, test_dir, batch_size = 64):
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
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

    # Crating the data loaders for train/test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader