"""
dataset.py

This module handles the preprocessing, loading, and batching of the ISIC 2020 dataset
for a Siamese network-based skin lesion classification task.

Author: Zain Al-Saffi
Date: 18th October 2024
"""

import os
import random
from collections import OrderedDict
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import logging
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_dataset(csv_file, img_dir, output_dir):
    """
    Preprocesses the ISIC 2020 dataset by organizing images into 'benign' and 'malignant' directories.

    Args:
        csv_file (str): Path to the CSV file containing image metadata.
        img_dir (str): Directory containing the original images.
        output_dir (str): Directory where the organized dataset will be saved.
    """
    df = pd.read_csv(csv_file)
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_name = row['isic_id'] + '.jpg'
        src_path = os.path.join(img_dir, img_name)
        
        if row['target'] == 0:  # Benign
            dst_path = os.path.join(benign_dir, img_name)
        else:  # Malignant
            dst_path = os.path.join(malignant_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            logging.warning(f"Image {src_path} not found.")

    logging.info(f"Preprocessing complete. Images organized in {output_dir}")

class ISIC2020Dataset(Dataset):
    """
    Custom Dataset class for the ISIC 2020 dataset.
    Handles loading of images, creation of triplets, and oversampling of minority class.

    Args:
        data_dir (str): Root directory containing the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        mode (str): 'train' for training set, 'test' for validation/test set.
        split_ratio (float): Ratio for splitting data into train and validation sets.
        oversample_factor (int, optional): Factor by which to oversample the minority class.
    """
    def __init__(self, data_dir, transform=None, mode='train', split_ratio=0.7, oversample_factor=None):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # Define paths for benign and malignant images
        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')

        # Load image paths and labels
        benign_images = self.get_image_paths(self.benign_dir, 0)
        malignant_images = self.get_image_paths(self.malignant_dir, 1)

        logging.info(f"Found {len(benign_images)} benign images and {len(malignant_images)} malignant images")

        if not benign_images or not malignant_images:
            raise ValueError("Insufficient images found in one or both classes.")

        # Combine benign and malignant images
        all_images = benign_images + malignant_images

        # Split into train and validation sets
        train_images, val_images = train_test_split(
            all_images,
            test_size=1 - split_ratio,
            stratify=[label for _, label in all_images],
            random_state=42
        )

        if mode == 'train':
            # Oversample malignant class in training set
            if oversample_factor is None:
                # Calculate oversample_factor to achieve desired balance (e.g., 60% benign, 40% malignant)
                benign_count = len([x for x in train_images if x[1] == 0])
                malignant_count = len([x for x in train_images if x[1] == 1])
                desired_malignant_count = int(benign_count * (50 / 50))  # Adjust the ratio as needed
                oversample_factor = max(desired_malignant_count // malignant_count, 1)
            self.images = self.oversample_minority_class(train_images, oversample_factor)
        else:
            # Use the unbalanced validation set as is
            self.images = val_images

        self.image_paths = [img_path for img_path, _ in self.images]
        self.labels = [label for _, label in self.images]

        # Create dictionaries to store indices for each class
        self.class_to_indices = {
            0: [i for i, label in enumerate(self.labels) if label == 0],
            1: [i for i, label in enumerate(self.labels) if label == 1]
        }

    def get_image_paths(self, directory, label):
        """
        Helper method to get all valid image paths from a directory.

        Args:
            directory (str): Path to the directory containing images.
            label (int): Label to assign to all images in this directory.

        Returns:
            list: List of tuples (image_path, label) for all valid images in the directory.
        """
        image_paths = []
        for img in os.listdir(directory):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, img)
                if os.path.isfile(img_path) and os.path.getsize(img_path) > 0:
                    image_paths.append((img_path, label))
                else:
                    logging.warning(f"Skipping invalid or empty file: {img_path}")
        return image_paths

    def oversample_minority_class(self, images, factor):
        """
        Oversample the minority class (malignant) to balance the dataset.

        Args:
            images (list): List of tuples (image_path, label) for all images.
            factor (int): Factor by which to oversample the minority class.

        Returns:
            list: Balanced list of tuples (image_path, label) with oversampled minority class.
        """
        benign = [img for img in images if img[1] == 0]
        malignant = [img for img in images if img[1] == 1]
        
        oversampled_malignant = malignant * factor
        
        combined = benign + oversampled_malignant
        random.shuffle(combined)
        
        return combined

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Fetch a triplet (anchor, positive, negative) of images.
        The anchor and positive are from the same class, while the negative is from a different class.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: (anchor, positive, negative, label) where anchor, positive, and negative are PIL Images
                   and label is an integer.
        """
        img_path, label = self.images[idx]
        anchor = self.load_image(img_path)

        # Get positive (same class) and negative (different class) samples
        positive_idx = random.choice([i for i in self.class_to_indices[label] if i != idx])
        negative_idx = random.choice(self.class_to_indices[1 - label])

        positive = self.load_image(self.image_paths[positive_idx])
        negative = self.load_image(self.image_paths[negative_idx])

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative, label

    def load_image(self, img_path):
        """
        Load an image from a file path, handling potential errors.

        Args:
            img_path (str): Path to the image file.

        Returns:
            PIL.Image: Loaded image, or a blank image if loading fails.
        """
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (224, 224), color='gray')  # Return a blank image in case of error

class BalancedBatchSampler(Sampler):
    """
    Sampler that yields a mini-batch of indices balanced between classes.
    This helps to handle class imbalance during training.

    Args:
        dataset (ISIC2020Dataset): The dataset to sample from.
        batch_size (int): Number of samples per batch.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.minority_indices = dataset.class_to_indices[1]  # Malignant
        self.majority_indices = dataset.class_to_indices[0]  # Benign

        self.minority_batch_size = self.batch_size // 2
        self.majority_batch_size = self.batch_size - self.minority_batch_size

        self.num_batches = min(len(self.minority_indices) // self.minority_batch_size,
                               len(self.majority_indices) // self.majority_batch_size)

    def __iter__(self):
        """Yield balanced batches of indices."""
        for _ in range(self.num_batches):
            minority_batch = np.random.choice(self.minority_indices, self.minority_batch_size, replace=False)
            majority_batch = np.random.choice(self.majority_indices, self.majority_batch_size, replace=False)
            batch = np.concatenate([minority_batch, majority_batch])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        """Return the number of batches per epoch."""
        return self.num_batches

def get_data_loaders(data_dir, batch_size=32, split_ratio=0.8, num_workers=4):
    """
    Create and return data loaders for training and validation.
    Applies appropriate transformations and uses BalancedBatchSampler for training.

    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Number of samples per batch.
        split_ratio (float): Ratio for splitting data into train and validation sets.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader) containing the DataLoader objects for training and validation.
    """
    # Define transformations for training data (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Define transformations for validation data (without augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # Create dataset instances
    train_dataset = ISIC2020Dataset(data_dir, transform=train_transform, mode='train', split_ratio=split_ratio)
    val_dataset = ISIC2020Dataset(data_dir, transform=val_transform, mode='test', split_ratio=split_ratio)

    # Create balanced sampler for training data
    train_sampler = BalancedBatchSampler(train_dataset, batch_size)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,  # Use standard batching for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

if __name__ == '__main__':
    # Example usage and dataset statistics
    csv_file = 'data/train-metadata.csv'
    img_dir = 'data/train-image/image'
    output_dir = 'preprocessed_data'
    
    # Uncomment the following line if you need to preprocess the dataset (do it in the first run only)
    preprocess_dataset(csv_file, img_dir, output_dir)
    
    data_dir = 'preprocessed_data'
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=32)
    
    # Print overall dataset statistics
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    #Checking if we balanced the dataset for real
    train_benign_count = sum(1 for label in train_dataset.labels if label == 0)
    train_malignant_count = sum(1 for label in train_dataset.labels if label == 1)
    val_benign_count = sum(1 for label in val_dataset.labels if label == 0)
    val_malignant_count = sum(1 for label in val_dataset.labels if label == 1)

    print("\nOverall Dataset Statistics:")
    print(f"Training set - Total: {len(train_dataset)}, Benign: {train_benign_count}, Malignant: {train_malignant_count}")
    print(f"Validation set - Total: {len(val_dataset)}, Benign: {val_benign_count}, Malignant: {val_malignant_count}")