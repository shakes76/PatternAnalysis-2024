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
    scikit-learn
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import utils


class ProstateDataset(Dataset):
    """ 
    Custom dataset used to store the Prostate data from Nifti images.
    
    Modified from Pytorch tutorial:
    https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 

    Attributes:
        img_dir (str): Directory containing the images (keras_slices)
        mask_dir (str): Directory with the masks (keras_slices_seg)
        early_stop (bool): Boolean flag to prematurely stop loading files
        plotting (bool): Boolean flag to visualise the images and masks
    
    Methods:
        img_show: Plot 6 of the images and their masks
        calculate_class_weights: Returns weights based on label frequency
    """

    def __init__(self, 
            img_dir: str, 
            mask_dir: str, 
            early_stop = False, 
            plotting = False
            ):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Load images and masks
        self.imgs = utils.load_data_2D_from_directory(
            img_dir, norm_image=True, resized=True, resizing_masks=False, early_stop=early_stop)
        self.masks = utils.load_data_2D_from_directory(
            mask_dir, norm_image=False, resized=True, resizing_masks=True,
            one_hot=True, early_stop=early_stop, dtype=np.float16)
        # Plotting requires segment data that is not one hot encoded
        if plotting:
            self.masks_plot = utils.load_data_2D_from_directory(
                mask_dir, norm_image=False, resized=False, one_hot=False, early_stop=early_stop)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]

        # Move channels to 0th index and convert np to tensor
        mask = np.moveaxis(mask, 2, 0) # (H, W, C) to (C, H, W)
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
            None, plots the images and masks
        """

        fig = plt.figure(figsize=(8, 8))
        cols, rows = 4, 3
        count = 0 # Variable to move through subplots
        for idx in range(start_idx, start_idx + (cols * rows) // 2):
            img = self.imgs[idx]
            mask = self.masks_plot[idx]
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.axis("off")
            count += 1
            fig.add_subplot(rows, cols, count)
            plt.imshow(mask.squeeze(), cmap='gray')
            plt.axis("off")
        plt.show()

    def calculate_class_weights(self) -> np.ndarray:
        """
        Calculates weights to be used in the loss criterion based on
        the appearance of each label using a subsample of the masks

        Parameters:
            None

        Returns:
            np.ndarray: Weights for each label
        """

        # Must load mask data that has been resized but not one hot encoded
        self.masks_single_channel = utils.load_data_2D_from_directory(
                self.mask_dir, norm_image=False, resized=True, resizing_masks=True, one_hot=False, early_stop=False)
        # Slice the big array randomly to calculate the weights
        np.random.shuffle(self.masks_single_channel)
        sliced = self.masks_single_channel[:1000]
        # Calculate class weights for criterion
        unique_classes = np.unique(sliced)
        weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=sliced.flatten())
        return weights


# For testing purposes
if __name__ == "__main__":
    # Image directories
    main_dir = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data')
    train_image_path = os.path.join(main_dir, 'keras_slices_train')
    train_mask_path = os.path.join(main_dir, 'keras_slices_seg_train')
    test_image_path = os.path.join(main_dir, 'keras_slices_test')
    test_mask_path = os.path.join(main_dir, 'keras_slices_seg_test')

    # Creating training dataset that can plot some of the data
    train_dataset = ProstateDataset(
        train_image_path, train_mask_path, early_stop=True, plotting=True)
    
    # Print statements for debugging and improving understanding
    print(len(train_dataset))
    print(type(train_dataset.imgs))
    print(train_dataset.imgs.shape)
    print(train_dataset.masks.shape)
    print(type(train_dataset[0][1]))
    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)

    # Plot the training images from index 15-20 inclusive
    train_dataset.img_show(15)

    # Calculate class weights
    print(train_dataset.calculate_class_weights())