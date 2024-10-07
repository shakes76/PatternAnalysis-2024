# dataset.py

import os
import random
from collections import OrderedDict
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_dataset(csv_file, img_dir, output_dir):
    """
    Preprocesses the ISIC 2020 dataset by organizing images into 'benign' and 'malignant' directories.

    Args:
        csv_file (str): Path to the CSV file containing image labels.
        img_dir (str): Path to the directory containing images.
        output_dir (str): Path to the output directory where preprocessed data will be saved.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create output directories
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    # Process each image
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_name = row['image_name'] + '.jpg'
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
    def __init__(self, data_dir, transform=None, mode='train', split_ratio=0.8):
        """
        Initializes the ISIC2020Dataset.

        Args:
            data_dir (str): Path to the directory containing preprocessed images.
            transform (callable, optional): Transformation to be applied on images.
            mode (str): 'train' or 'test' mode.
            split_ratio (float): Ratio of data to be used for training.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        self.benign_dir = os.path.join(data_dir, 'benign')
        self.malignant_dir = os.path.join(data_dir, 'malignant')

        benign_images = self.get_image_paths(self.benign_dir, 0)
        malignant_images = self.get_image_paths(self.malignant_dir, 1)

        all_images = benign_images + malignant_images

        logging.info(f"Found {len(benign_images)} benign images and {len(malignant_images)} malignant images")

        if not all_images:
            raise ValueError("No images found in the dataset.")

        # Stratified split
        train_images, test_images = train_test_split(
            all_images,
            test_size=1 - split_ratio,
            stratify=[x[1] for x in all_images],
            random_state=42
        )

        self.images = train_images if mode == 'train' else test_images
        self.image_paths = [img_path for img_path, _ in self.images]
        self.labels = [label for _, label in self.images]
        logging.info(f"Using {len(self.images)} images for {mode}")

        self.class_to_indices = {
            0: [i for i, label in enumerate(self.labels) if label == 0],
            1: [i for i, label in enumerate(self.labels) if label == 1]
        }

        # Preload images into memory to speed up training (optional)
        # You can comment this out if you have memory constraints
        self.preload_images = False
        if self.preload_images:
            self.preloaded_images = {}
            for idx in tqdm(range(len(self)), desc="Preloading images"):
                img_path = self.image_paths[idx]
                image = self.load_image(img_path)
                self.preloaded_images[img_path] = image

    def get_image_paths(self, directory, label):
        image_paths = []
        for img in os.listdir(directory):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(directory, img)
                if os.path.isfile(img_path) and os.path.getsize(img_path) > 0:
                    image_paths.append((img_path, label))
                else:
                    logging.warning(f"Skipping invalid or empty file: {img_path}")
        return image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        anchor_img_path = self.image_paths[idx]
        anchor_label = self.labels[idx]

        anchor = self.get_image(anchor_img_path)

        # Get positive and negative samples
        positive = self.get_random_image(anchor_label, exclude_idx=idx)
        negative = self.get_random_image(1 - anchor_label)

        return anchor, positive, negative, anchor_label

    def get_image(self, img_path):
        if self.preload_images:
            image = self.preloaded_images[img_path]
        else:
            image = self.load_image(img_path)

        if self.transform:
            image = self.transform(image)
        return image

    def load_image(self, img_path):
        try:
            image = Image.open(img_path).convert('RGB')
            return image
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            return Image.new('RGB', (224, 224), color='gray')  # Return a blank image in case of error

    def get_random_image(self, label, exclude_idx=None):
        indices = self.class_to_indices[label]
        if exclude_idx is not None:
            indices = [idx for idx in indices if idx != exclude_idx]

        if not indices:
            logging.warning(f"No images found for label {label}. Using a fallback image.")
            return Image.new('RGB', (224, 224), color='gray')

        random_idx = random.choice(indices)
        img_path = self.image_paths[random_idx]
        return self.get_image(img_path)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, oversample_ratio=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size
        self.oversample_ratio = oversample_ratio

        class_counts = {
            0: len(self.dataset.class_to_indices[0]),
            1: len(self.dataset.class_to_indices[1])
        }
        self.majority_class = max(class_counts, key=class_counts.get)
        self.minority_class = min(class_counts, key=class_counts.get)

        total_samples = batch_size
        minority_samples = total_samples // (oversample_ratio + 1)
        majority_samples = total_samples - minority_samples

        self.samples_per_class = {
            self.majority_class: majority_samples,
            self.minority_class: minority_samples
        }

        # Initialize counters
        self.benign_count = 0
        self.malignant_count = 0

    def __iter__(self):
        # Reset counters at the start of each epoch
        self.benign_count = 0
        self.malignant_count = 0

        for _ in range(self.num_batches):
            batch = []
            for class_label, num_samples in self.samples_per_class.items():
                indices = self.dataset.class_to_indices[class_label]
                selected_indices = random.choices(indices, k=num_samples)
                batch.extend(selected_indices)
                
                # Update counters
                if class_label == 0:  # Benign
                    self.benign_count += num_samples
                else:  # Malignant
                    self.malignant_count += num_samples

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

    def get_sample_counts(self):
        return {
            "benign": self.benign_count,
            "malignant": self.malignant_count
        }

def get_data_loaders(data_dir, batch_size=32, split_ratio=0.8, num_workers=4):
    """
    Creates data loaders for training and validation datasets.

    Args:
        data_dir (str): Path to the preprocessed data directory.
        batch_size (int): Batch size for data loaders.
        split_ratio (float): Ratio of data to use for training.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ISIC2020Dataset(data_dir, transform=train_transform, mode='train', split_ratio=split_ratio)
    val_dataset = ISIC2020Dataset(data_dir, transform=val_transform, mode='test', split_ratio=split_ratio)

    # Count benign and malignant samples in the training dataset
    benign_count = sum(1 for label in train_dataset.labels if label == 0)
    malignant_count = sum(1 for label in train_dataset.labels if label == 1)

    print(f"Training dataset composition after augmentation:")
    print(f"Benign samples: {benign_count}")
    print(f"Malignant samples: {malignant_count}")
    print(f"Total samples: {len(train_dataset)}")

    train_sampler = BalancedBatchSampler(train_dataset, batch_size)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

if __name__ == '__main__':
    # Test the data loading
    csv_file = 'ISIC_2020_Training_GroundTruth_v2.csv'
    img_dir = 'data/ISIC_2020_Training_JPEG/train/'
    output_dir = 'preprocessed_data'
    
    preprocess_dataset(csv_file, img_dir, output_dir)
    data_dir = 'preprocessed_data'
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=32)
    
    for i, (anchor, positive, negative, labels) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"Anchor shape: {anchor.shape}")
        print(f"Positive shape: {positive.shape}")
        print(f"Negative shape: {negative.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        
        if i == 2:
            break
