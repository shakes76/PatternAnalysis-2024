import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_metric_learning.samplers import MPerClassSampler

LOCAL = True  # For my local machine
IMAGE_DIR = os.path.expanduser('~/Projects/COMP3710/siamese_project/dataset/train-image/image/') if not LOCAL else \
    os.path.expanduser('~/.kaggle/datasets/isic-2020-jpg-256x256-resized/train-image/image/')
ANOT_FILE = os.path.expanduser('~/Projects/COMP3710/siamese_project/dataset/train-metadata.csv') if not LOCAL else \
    os.path.expanduser('~/.kaggle/datasets/isic-2020-jpg-256x256-resized/train-metadata.csv')


class ISICKaggleDataset(Dataset):
    def __init__(self, annotations_file, img_dir, indices=None, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        if indices is not None:
            self.img_labels = self.img_labels.iloc[indices].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Return an image and its label.
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]) + '.jpg'
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        label = self.img_labels.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, label


def split_dataset(metadata, test_size=0.2, val_size=0.1, random_state=42):
    """
    Splits dataset into train, validation, and test sets using stratified sampling.
    """
    labels = metadata['target']
    train_indices, test_indices = train_test_split(range(len(metadata)), test_size=test_size, stratify=labels, random_state=random_state)
    train_labels = labels.iloc[train_indices]
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (1 - test_size), stratify=train_labels,
                                                  random_state=random_state)

    return train_indices, val_indices, test_indices


def get_data_loaders(batch_size=32, sampler=None):
    """
    Prepares and returns the data loaders for training, validation, and test datasets using a 70/20/10 split.
    Allows for the use of a sampler for the training dataset.
    """
    # Load metadata from CSV
    metadata = pd.read_csv(ANOT_FILE)

    # Split the dataset into train, val, and test indices using stratified sampling
    train_indices, val_indices, test_indices = split_dataset(metadata)

    # Define transformations for the training set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=(224, 224)),
    ])

    # Create datasets for each split
    train_dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, indices=train_indices, transform=train_transform)
    val_dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, indices=val_indices)
    test_dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, indices=test_indices)

    # Create data loaders for training, validation, and test sets
    if sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def show_image_grid(dataset, num_images=5):
    """
    Displays a grid of images from the dataset.
    """
    fig, ax = plt.subplots(1, num_images, figsize=(20, 5))

    for i in range(num_images):
        image, label = dataset[i]
        ax[i].imshow(image.permute(1, 2, 0))
        ax[i].set_title(f"Label: {label}")
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()
