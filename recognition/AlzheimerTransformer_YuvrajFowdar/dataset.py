"""
DOCSTRING ABOUT WHAT THIS FILE IS HERE 

"""
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

def get_mean_std(loader: DataLoader):
    mean = torch.zeros(3)
    squared_mean = torch.zeros(3)
    N = 0 # Number of batches
    for images, _ in tqdm(loader, desc="Computing mean and std"): 
        # Images are [32, 3, 224, 224]
        
        num_batches, num_channels, height, width = images.shape
        N += num_batches

        mean += images.sum(dim=(0,2,3)) # Mean is size [3], i.e it's the mean sum over each channel [R, G, B]...
        squared_mean += (images ** 2).sum(dim=(0,2,3)) # Accumulate squared mean
    mean /= N * height * width # Divide the summed mean by the number of pixels we've ever seen (i.e mean is on a per pixel basis)

    squared_mean /= N * height * width # Same with squared mean
    
    # Get std
    std = torch.sqrt((squared_mean - mean ** 2)) # Std per pixel
    
    return mean, std





