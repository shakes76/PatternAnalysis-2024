import argparse
import os
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import numpy as np
from tqdm import tqdm
import utils
import glob


batch_size = 10

def get_dataloaders(batch_size): 
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    train_file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
    test_file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

    train_files = sorted(glob.glob(f"{train_file_path}/**.nii.gz", recursive = True))
    test_files = sorted(glob.glob(f"{test_file_path}/**.nii.gz", recursive = True))

    train_dataset = utils.load_data_2D(train_files[0:64])
    test_dataset = utils.load_data_2D(test_files[0:64])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def plot_from_dataloader(batch_size: int, dataloader):
    real_batch = next(iter(dataloader))

    fig, axes = plt.subplots(1, batch_size , figsize=(15, 15))
    axes = axes.flatten()
    plt.title("Test1")
    for i in range(10):
        img = real_batch[i, :, :].cpu().numpy() 
        axes[i].imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
        axes[i].axis('off')  

    plt.savefig("./Project/test1.png")

train_dataloader, test_dataloader = get_dataloaders(batch_size)
# plot_from_dataloader(batch_size, train_dataloader)