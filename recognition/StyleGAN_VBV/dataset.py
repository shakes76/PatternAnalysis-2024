import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set the dataset path
DATASET_PATH = '/home/groups/comp3710/ADNI'

def get_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader, dataset
