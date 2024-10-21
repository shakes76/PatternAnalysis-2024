import torch
import torch.nn.parallel
import torch.utils.data
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import *
import glob

def get_dataloaders(batch_size): 
    
    train_file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
    test_file_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"

    train_files = sorted(glob.glob(f"{train_file_path}/**.nii.gz", recursive = True))
    test_files = sorted(glob.glob(f"{test_file_path}/**.nii.gz", recursive = True))

    train_dataset = load_data_2D(train_files, normImage=True)
    test_dataset = load_data_2D(test_files)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def plot_from_dataloader(batch_size: int, dataloader, directory):
    real_batch = next(iter(dataloader))

    fig, axes = plt.subplots(1, batch_size , figsize=(15, 15))
    axes = axes.flatten()
    plt.title("Test1")
    for i in range(batch_size):
        img = real_batch[i, :, :].cpu().numpy() 
        axes[i].imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
        axes[i].axis('off')  

    plt.savefig(f"./Project/{directory}/training_img_samples.png")
