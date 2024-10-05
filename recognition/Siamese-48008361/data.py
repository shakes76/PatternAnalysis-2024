"""
This module provides functionality for loading and processing the ISIC 2020 dataset
for use in a Siamese network with triplet loss for melanoma classification.
"""
import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ISIC2020Dataset(Dataset):
    """
    A custom Dataset class for the ISIC 2020 skin lesion dataset.

    This class handles loading of images, data augmentation, and generation of triplets
    (anchor, positive, negative) for training a Siamese network with triplet loss.

    Attributes:
        data (pd.DataFrame): The dataset containing image information and labels.
        img_dir (str): Directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        mode (str): 'train' for training set, 'test' for test set.
        benign (pd.DataFrame): Subset of data containing only benign samples.
        malignant (pd.DataFrame): Subset of data containing only malignant samples.
    """


    def __init__(self, csv_file, img_dir, transform=None, mode='train', split_ratio=0.8):

        """
        Args:
            csv_file (str): Path to the csv file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (str): 'train' for training set, 'test' for test set.
            split_ratio (float): Ratio of data to use for training.
        """
        
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

        # Split data into train and test
        self.train_data, self.test_data = self.train_test_split(split_ratio)

        if mode == 'train':
            self.data = self.train_data
            # Augment minority class (malignant) to match majority class (benign)
            benign_count = len(self.data[self.data['target'] == 0])
            malignant_samples = self.data[self.data['target'] == 1]
            # Augment factor is the number of times we need to repeat the malignant samples
            augment_factor = benign_count // len(malignant_samples) - 1
            augmented_malignant = pd.concat([malignant_samples] * augment_factor, ignore_index=True)
            self.data = pd.concat([self.data, augmented_malignant], ignore_index=True)
        else:
            self.data = self.test_data
        # Create subsets for benign and malignant samples
        self.benign = self.data[self.data['target'] == 0]
        self.malignant = self.data[self.data['target'] == 1]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):

        """
        Generates one sample of data including a triplet (anchor, positive, negative).
        
        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (anchor_img, positive_img, negative_img, anchor_label)
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the anchor image
        anchor_row = self.data.iloc[idx]
        anchor_img = self.get_image(anchor_row)
        anchor_label = anchor_row['target']
        # Select positive and negative samples based on the anchor's label
        if anchor_label == 0:  # benign
            positive = self.get_image(self.benign.sample().iloc[0])
            negative = self.get_image(self.malignant.sample().iloc[0])
        else:  # malignant
            positive = self.get_image(self.malignant.sample().iloc[0])
            negative = self.get_image(self.benign.sample().iloc[0])

        return anchor_img, positive, negative, anchor_label
    
    def get_image(self, row):
        """
        Loads and processes an image.

        Args:
            row (pd.Series): A row from the dataset containing image information.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        img_name = row['image_name']
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Apply additional random augmentation for malignant samples in training mode
        if self.mode == 'train' and row['target'] == 1:
            image = self.random_augment(image)
        
        return image
