import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

'''
Gets the mean and standard deviations values from a dataloader
'''
def get_mean_std(dataloader):
    mean = 0.0
    mean_sqr = 0.0
    n = 0

    for inputs, _ in dataloader:
        n += inputs.size(0)
        mean += inputs.sum(dim=(0,2,3))
        mean_sqr += (inputs ** 2).sum(dim=(0,2,3))

    mean /= n * inputs.size(2) * inputs.size(3)
    mean_sqr /= n * inputs.size(2) * inputs.size(3)
    std = (mean_sqr - mean ** 2).sqrt()

    return mean, std 

""" Returns the train and test dataloaders for the ADNI dataset """
def get_dataloaders(batch_size=32, image_size=224, path="recognition/GFNet-47428364/AD_NC"):
    # Create transformer
    pre_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    pre_train_dataset = ImageFolder(root=path+"/train", transform=pre_transforms)
    sampler = SubsetRandomSampler(torch.randperm(len(pre_train_dataset))[:1000])
    sample_loader = DataLoader(pre_train_dataset, batch_size=batch_size, sampler=sampler)

    mean, std = get_mean_std(sample_loader)

    # Normalised transformations for training
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.15, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Normalised transformations for testing
    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Generate datasets
    train_dataset = ImageFolder(root=path+"/train", transform=train_transforms)
    test_dataset = ImageFolder(root=path+"/test", transform=test_transforms)
                              
    # Generate dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader