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
from functools import lru_cache


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
    
    #@lru_cache(maxsize=1000)
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
        
        if self.mode == 'train' and row['target'] == 1:
            image = self.random_augment(image)
        
        return image

    
    def random_augment(self, image):
        """
        Applies random augmentations to an image.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Augmented image tensor.
        """
        # Define a set of possible augmentations
        augmentations = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        # Randomly apply some of these augmentations
        for aug in augmentations:
            if random.random() > 0.5:
                image = aug(image)
        return image

    def train_test_split(self, split_ratio):
        """
        Splits the data into training and testing sets.

        Args:
            split_ratio (float): Ratio of data to use for training.

        Returns:
            tuple: (train_data, test_data)
        """
        train_data, test_data = train_test_split(
            self.data, 
            test_size=1-split_ratio, 
            stratify=self.data['target'],
            random_state=42
        )
        return train_data, test_data
    
def get_data_loaders(csv_file, img_dir, batch_size=32, split_ratio=0.8):
    """
    Creates DataLoader objects for both training and testing datasets.

    Args:
        csv_file (str): Path to the csv file with annotations.
        img_dir (str): Directory with all the images.
        batch_size (int): How many samples per batch to load.
        split_ratio (float): Ratio of data to use for training.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to a standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing with ImageNet stats
    ])

    # Create datasets
    train_dataset = ISIC2020Dataset(csv_file, img_dir, transform=transform, mode='train', split_ratio=split_ratio)
    test_dataset = ISIC2020Dataset(csv_file, img_dir, transform=transform, mode='test', split_ratio=split_ratio)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

    return train_loader, test_loader

