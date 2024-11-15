"""
File: dataset.py
Description: Contains dataset classes for loading and preparing image data for training a Siamese Network.
             The ISICDataset class loads images and applies augmentations, especially for malignant samples.
             The SiameseDataset class prepares paired images with labels for similarity learning.
             
Classes:
    ISICDataset: Loads the ISIC dataset with optional augmentations for malignant images.
    SiameseDataset: Generates pairs of images (similar or different) for training the Siamese Network.
"""

import os
import random
import pandas as pd
import kagglehub
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class ISICDataset(Dataset):
    """
    ISICDataset class to load and preprocess the ISIC skin cancer dataset.
    Augments malignant images to balance the dataset.

    Args:
        dataset_path (str): Path to the directory containing image files.
        metadata_path (str): Path to the CSV file with image labels.
        transform (torchvision.transforms.Compose, optional): Transformations to apply to images.
        augment_transform (torchvision.transforms.Compose, optional): Transformations for augmenting malignant images.
        num_augmentations (int, optional): Number of augmentations to apply to each malignant image.
    """
    def __init__(self, dataset_path, metadata_path, transform=None, augment_transform=None, num_augmentations=5):
        self.dataset_path = dataset_path
        self.transform = transform
        self.augment_transform = augment_transform
        self.num_augmentations = num_augmentations
        self.labels = self.load_labels(metadata_path)  # Load image labels
        self.data, self.malignant_data = self.load_data()  # Load image paths

    def load_labels(self, metadata_path):
        """
        Load labels from the metadata file.

        Args:
            metadata_path (str): Path to the CSV file with image labels.

        Returns:
            dict: A dictionary mapping image filenames to labels (1 for malignant, 0 for benign).
        """
        metadata = pd.read_csv(metadata_path)
        return {row['isic_id'] + '.jpg': row['target'] for _, row in metadata.iterrows()}

    def load_data(self):
        """
        Load image filenames and separate malignant images for augmentation.

        Returns:
            tuple: A list of all images and a list of malignant images.
        """
        image_files = []
        malignant_files = []
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if (file.endswith('.jpg') or file.endswith('.jpeg')) and file in self.labels:
                    image_files.append(file)
                    if self.labels[file] == 1:  # Label 1 indicates malignant
                        malignant_files.append(file)

        return image_files, malignant_files

    def __len__(self):
        """
        Calculate the total dataset length, including augmented images.

        Returns:
            int: Total count of original and augmented images.
        """
        return len(self.data) + (self.num_augmentations * len(self.malignant_data))

    def __getitem__(self, index):
        """
        Load an image and its label, applying augmentations if it's a malignant case.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            tuple: The transformed image and its label.
        """
        if index < len(self.data):
            # Load original image if within range of original data
            img_name = self.data[index]
            augmentation = False
        else:
            # Augment malignant image if index is beyond original data
            aug_index = index - len(self.data)
            original_index = aug_index // self.num_augmentations
            augmentation = True
            img_name = self.malignant_data[original_index]

        img_path = os.path.join(self.dataset_path, img_name)
        img = Image.open(img_path).convert('RGB')  # Convert image to RGB

        if augmentation and self.augment_transform:
            img = self.augment_transform(img)  # Apply augmentations if necessary

        if self.transform:
            img = self.transform(img)  # Apply standard transformations

        label = self.labels[img_name]  # Get label for the image

        return img, label


class SiameseDataset(Dataset):
    """
    SiameseDataset class to create pairs of images for training a Siamese Network.

    Args:
        dataset (Dataset): The dataset of single images to pair.
        num_pairs (int, optional): Number of pairs to generate (default is 50000).
        transform (torchvision.transforms.Compose, optional): Transformations to apply to both images in a pair.
    """
    def __init__(self, dataset, num_pairs=50000, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.num_pairs = num_pairs

    def generate_pair(self, index):
        """
        Generate a pair of images, either from the same or different classes.

        Args:
            index (int): Index used to select and pair images.

        Returns:
            tuple: Two images, and a similarity label (1 for dissimilar, 0 for similar).
        """
        index = index % len(self.dataset)
        img0, label0 = self.dataset[index]  # Get base image and label
        should_get_same_class = random.randint(0, 1)  # Randomly choose pair type

        if should_get_same_class:
            # Positive pair: Select another image with the same label
            while True:
                img1, label1 = random.choice(self.dataset)
                if label0 == label1:
                    break
        else:
            # Negative pair: Select an image with a different label
            while True:
                img1, label1 = random.choice(self.dataset)
                if label0 != label1:
                    break

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        similarity_label = torch.tensor(int(label0 != label1), dtype=torch.float32)  # 1 for dissimilar, 0 for similar

        return img0, img1, similarity_label

    def __len__(self):
        """
        Returns the number of image pairs.

        Returns:
            int: Total number of pairs to generate.
        """
        return self.num_pairs

    def __getitem__(self, index):
        """
        Get a generated image pair and similarity label.

        Args:
            index (int): Index of the pair to retrieve.

        Returns:
            tuple: Two images and a similarity label.
        """
        return self.generate_pair(index)

if __name__ == "__main__":
    # Download the ISIC dataset from Kaggle and set paths
    dataset_path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
    dataset_image_path = os.path.join(dataset_path, "train-image/image")
    meta_data_path = os.path.join(dataset_path, "train-metadata.csv")

    # Define standard transformations for resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Define augmentation transformations for malignant cases
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ])

    # Instantiate the ISIC dataset with transformations and augmentations
    isic_dataset = ISICDataset(
        dataset_path=dataset_image_path,
        metadata_path=meta_data_path,
        transform=transform,
        augment_transform=augment_transform,
        num_augmentations=5  # Number of augmentations per malignant case
    )
