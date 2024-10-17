import numpy as np
import torch
import tensorflow as tf
from torchvision import  datasets, transforms
import torchvision

def data_set_creator():
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((256, 256)),
        transforms.ToTensor()
    ])


    data_dir = 'recognition/Stable Diffusion - 47219647/AD_NC/test'
    dataset = datasets.ImageFolder(root=data_dir, transform=augmentation_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)
    
    return data_loader


