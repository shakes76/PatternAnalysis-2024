""" 
This file handles the dataset used for the Prostate data.
A custom PyTorch dataset class called ProstateDataset has been made
and uses utils.py to load 2D Nifti image files from the specified
directory. It can also visualise those images.

Author: 
    Joseph Reid

Classes:
    ProstateDataset: Dataset to store Prostate images for loading

Dependencies:
    numpy
    pytorch
    matplotlib
"""

import os
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class ProstateDataset(Dataset):
    """ 
    Custom dataset used to store the Prostate data from Nifti images.
    
    Modified from Pytorch tutorial:
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 

    Attributes:
        img_dir (str): Directory containing the images (keras_slices)
        mask_dir (str): Directory with the masks (keras_slices_seg)
        early_stop (bool): Boolean flag to only load some of the images
        plotting (bool): Boolean flag to visualise the images and masks
    
    Methods:
        img_show: Plot 6 of the images and their masks
    """

    def __init__(self, img_dir: str, mask_dir: str, early_stop = True, plotting = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir

        # Load images and masks
        self.imgs = utils.load_data_2D_from_directory(self.img_dir, early_stop=early_stop)
        self.masks = utils.load_data_2D_from_directory(self.mask_dir, early_stop=early_stop, categorical=True)

        # Plotting requires segment data that is not one hot encoded
        if plotting:
            self.masks_for_plotting = utils.load_data_2D_from_directory(self.mask_dir, early_stop=early_stop, categorical=False)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        # Move channels to 0th index and convert np to tensor
        mask = np.moveaxis(mask, 2, 0)
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
    
        return (img, mask)
    
    def img_show(self, start_idx: int = 0):
        """
        Plots 6 of the image files, and their corresponding masks,
        starting from the specified index.
        
        Modified from Pytorch tutorial: 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

        Parameters:
            start_idx (int): Image index that should be plotted from

        Returns:
            None
        """

        fig = plt.figure(figsize=(8, 8))
        cols, rows = 4, 3
        count = 0 # Variable to move through subplots
        for idx in range(start_idx, start_idx + (cols * rows) // 2):
            img = self.imgs[idx]
            mask = self.masks_for_plotting[idx]
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis("off")
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(mask.squeeze(), cmap='gray')
            plt.axis("off")
        plt.show()

# For testing purposes
if __name__ == "__main__":
    # Image directories
    main_dir = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data')
    train_image_path = os.path.join(main_dir, 'keras_slices_train')
    train_mask_path = os.path.join(main_dir, 'keras_slices_seg_train')
    test_image_path = os.path.join(main_dir, 'keras_slices_test')
    test_mask_path = os.path.join(main_dir, 'keras_slices_seg_test')

    # Creating training dataset that can plot some of the data
    train_dataset = ProstateDataset(train_image_path, train_mask_path, plotting=True)
    
    # Print statements for debugging and improving understanding
    print(len(train_dataset))
    print(type(train_dataset[1][1]))
    print(train_dataset[1][0].shape)
    print(train_dataset[1][1].shape)

    # Plot the training images from index 15-20 inclusive
    train_dataset.img_show(15)