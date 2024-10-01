"""
For loading the ADNI dataset and pre processing the data
"""

import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import numpy as np
import torchvision

if __name__ == "__main__":

    # The path when running locally
    data_directory = '../../../AD_NC'
    #the path to the directory on Rangpur
    #data_directory = '/home/groups/comp3710/ADNI/AD_NC'

    #Set Hyperparameters
    image_size = 256
    batch_size = 32

    # First Configure the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not found, using CPU")
    print(torch.cuda.get_device_name(0))

    print("Start DataLoading ...")


    # Transformations applied during data loading
    transform_train=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    transform_test=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    transform_val=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])

    # Create the training and testing datasets using ImageFolder
    train_dataset = ImageFolder(root=os.path.join(data_directory , 'train'), transform=transform_train)
    test_dataset = ImageFolder(root=os.path.join(data_directory, 'test'), transform=transform_test)

    # Create the DataLoader for train, test
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Print a few statistics about the dataset
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of testing images: {len(test_dataset)}")

    # Checking the dataset classes (AD and NC)
    print(f"Classes in train dataset: {train_dataset.classes}")
    print(f"Classes in test dataset: {test_dataset.classes}")

