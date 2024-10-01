"""
Extra functions and classes that have been used to create this model.

This code does not been to be included to run the actual model. Much of the processing here
has been hardcoded into the model.

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
import dataset

image_size = 256
batch_size = 32

# The path when running locally
data_directory = os.path.join('../../../AD_NC')
#the path to the directory on Rangpur
#data_directory = '/home/groups/comp3710/ADNI/AD_NC'

if __name__ == "__main__":
    transform_no_norm ={ 'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()  # Only tensor conversion, no normalization
    ])
}
    # Create train, test, and validation datasets
    train_dataset = dataset.ADNIDataset(data_dir=os.path.join(data_directory, 'train'), transform=transform_no_norm, mode='train')
    # DataLoader for batching
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1)

    # Get the standard deviation and means for the normalisation in dataset.py
    def get_mean_std(dataset):
        '''Compute the mean and std value of dataset.'''
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:,i,:,:].mean()
                std[i] += inputs[:,i,:,:].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std

    # Calculate mean and std
    mean, std = get_mean_std(train_dataset)
    print(f"Calculated mean: {mean}")
    print(f"Calculated std: {std}")
