"""
Contains functions to initialise the DataLoaders for the train, test, and validation data sets.
Plots a set of images from a Dataloader. 

Author: Kirra Fletcher s4745168
"""
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import *
import glob


def get_dataloaders(batch_size, train_file_path, test_file_path): 
    """
    Creates dataloaders for the training, test, and validation datasets with the desired batch size.
    All three datasets are normalised. Training data is shuffled, test and validation data are not shuffled.

    Parameters:
        batch_size: Batch size for the dataloaders
        train_file_path: Directory to the training Nifti files
        test_file_path: Directory to the test Nifti files

    Returns:
        Train, test, and validation DataLoaders respectively
    """

    # In comp3710 directory, train size is 11460, test is 540
    train_files = sorted(glob.glob(f"{train_file_path}/**.nii.gz", recursive = True))
    test_files = sorted(glob.glob(f"{test_file_path}/**.nii.gz", recursive = True))

    # Validation set size = twice test set size, i.e. 540*2 = 1080
    # Validation set removed from training set
    validation_files = random.sample(train_files, len(test_files) * 2)
    for file in validation_files:
        train_files.remove(file)

    train_dataset = load_data_2D(train_files, normImage=True)
    test_dataset = load_data_2D(test_files, normImage=True)
    validation_dataset = load_data_2D(validation_files, normImage=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader, validation_dataloader

def plot_from_dataloader(dataloader, directory):
    """
    Plots a batch of images from the given dataloader.
    Saves the plotted images to the given directory.
    """
    real_batch = next(iter(dataloader))
    batch_size = len(real_batch)

    fig, axes = plt.subplots(1, batch_size , figsize=(15, 15))
    axes = axes.flatten()
    plt.title("Original Images")
    for i in range(batch_size):
        img = real_batch[i, :, :].cpu().numpy() 
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')  

    plt.savefig(f"{directory}/training_img_samples.png")
