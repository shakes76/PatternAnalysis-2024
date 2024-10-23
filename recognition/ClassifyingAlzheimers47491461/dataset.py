import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Initial 256 width, 240 height
"""
Initial preprocessing of data
"""

def mean_std():
    # Converts image to 2d tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Loads from local dataset (only needs to run once regardless)
    dataset = datasets.ImageFolder(root='C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\test', transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=10)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_pixels = 0

    for images, _ in dataloader:

        batch_size, num_channels, height, width = images.shape
        num_pixels_in_batch = batch_size * height * width

        # Summing across all channels over every image
        mean += images.sum(dim=[0, 2, 3])
        std += (images ** 2).sum(dim=[0, 2, 3])

        total_pixels += num_pixels_in_batch
    # Calculates per-pixel average
    mean /= total_pixels
    # As BW channels will have same std and rgb average
    print(f"rgb mean{mean}")
    print(f"rgb stdv{torch.sqrt(std / total_pixels - mean ** 2)}")

def process(colab=False):
    preprocess = transforms.Compose([
        # tTransforming images to fixed 'square' size as done in research paper
        transforms.Resize((224, 224)),
        # Random cropping of images to prevent over-fitting and improve generalisation
        # This is also done in the research paper to "help the model generalise better"
        transforms.RandomResizedCrop(224),
        # Although probably not required (given nature of data), introduce some invariance
        # to left-right orientation (Also done in paper)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalisation based on the average RGB and std RGB values of training set
        transforms.Normalize(mean=[0.1167, 0.1167, 0.1167],
                             std=[0.2258, 0.2258, 0.2258]),
    ])
    if not colab:
        dataset = ImageFolder(root='C:\\Users\\User\\Desktop\\3710Aux\\ADNI_AD_NC_2D\\AD_NC\\train', transform=preprocess)
    else:
        dataset = ImageFolder(root='/content/drive/MyDrive/ADNI/train', transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10)
    return dataloader


if __name__ == "__main__":
    process(colab=False)
