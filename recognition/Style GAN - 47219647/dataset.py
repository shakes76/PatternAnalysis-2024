import numpy as np
import torch
import tensorflow as tf
from torchvision import  datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes)

def data_set_creator(batch_size):
    augmentation_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),  # Can dynamically resize based on step
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    # Update the directory as needed
    data_dir = 'recognition/Style GAN - 47219647/AD_NC/'
    
    dataset = datasets.ImageFolder(root=data_dir, transform=augmentation_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    return data_loader, dataset




