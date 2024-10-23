"""
Contains the data loaders for loading and preprocessing the ISIC 2020 Dataset.
This module will help prepare training, validation and testing data to be used 
by a siamese network using triplet loss.
"""

###############################################################################
### Imports
import os
import random
import numpy as np
import pandas as pd
import torch
import cv2

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms


###############################################################################
### Classes
class TripletDataGenerator(torch.utils.data.Dataset):
    """
    Dataset Generator designed to group data (a list of labels and images)
    into triplets (if required) for use in a triplet loss function.

    The 'images' should be supplied as a list of image paths, this class will read them into
    images when get item is called.

    Additionally if any  transform is specified - the data will be run through said
    transform before returning them.
    """
    def __init__(self, images: list, labels: list=None, train: bool=True, transform: callable=None) -> None:
        self.is_train = train
        self.transform = transform
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the number of images in the data set
        """
        return len(self.images)

    def __getitem__(self, anchor_idx: int) -> tuple[torch.tensor, torch.tensor, torch.tensor, int]:
        """
        Returns the next triplet in the data set.

        Each triplet is made up of an anchor data point, a positive data point (point of the same
        class as the anchor) and a negative data point (point of the opposite class to the anchor).
        The class of the anchor will also be returned
        """
        anchor_img = cv2.imread(self.images[anchor_idx]) / 255.0
        anchor_label = self.labels[anchor_idx]

        # If we are training we want the full triplet
        if self.is_train:
            # All images that are of the same class as anchor but not anchor
            positive_list = [idx for idx, label in enumerate(self.labels) if label == anchor_label and idx != anchor_idx]
            positive_idx = random.choice(positive_list)
            positive_img = cv2.imread(self.images[positive_idx]) / 255.0

            # All images that are of the different class then anchor and not anchor (This check is not required, but nice for consistancy)
            negative_list = [idx for idx, label in enumerate(self.labels) if label != anchor_label and idx != anchor_idx]
            negative_idx = random.choice(negative_list)
            negative_img = cv2.imread(self.images[negative_idx]) / 255.0

            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            return anchor_img, positive_img, negative_img, anchor_label

        # If we are testing, we only care about the anchor image, the other images
        # can can returned as empty tensors
        else:
            if self.transform:
                anchor_img = self.transform(anchor_img)
            return anchor_img, torch.empty(1), torch.empty(1), anchor_label


###############################################################################
### Functions
def get_isic2020_data(metadata_path, image_dir, data_subset: int | None=None):
    """
    Returns: images, labels
    """
    metadata = pd.read_csv(metadata_path)

    # Add the file extension to isic_id to match image filenames
    metadata['image_file'] = metadata['isic_id'] + '.jpg'

    # Map image filename to target class and get a list of the file paths
    image_to_label = dict(zip(metadata['image_file'], metadata['target']))
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img in image_to_label]

    # If we are using subset of the data, ensure that we try our best to have an equal number of each class
    if data_subset:
        pos_paths = [img for img in image_paths if image_to_label[os.path.basename(img)] == 1][:data_subset // 2]
        neg_paths = [img for img in image_paths if image_to_label[os.path.basename(img)] == 0][:data_subset // 2]
        image_paths = pos_paths + neg_paths

    # Get a list of the labels for the images we are using
    labels = [image_to_label[os.path.basename(path)] for path in image_paths]
    return np.array(image_paths), np.array(labels)

def train_val_test_split(images, labels):
    """
    0.8 0.1 0.1 split

    Stratified sampling
    With oversampling of the training data to ensure that the
    classes are balanced in training.
    
    Returns: train_images, val_images, test_images, train_labels, val_labels, test_labels
    """
    train_images, other_images, train_labels, other_labels = train_test_split(images, labels, test_size=0.2, stratify=labels)
    test_images, val_images, test_labels, val_labels = train_test_split(other_images, other_labels, test_size=0.5, stratify=other_labels)

    # Preform oversampling of training data
    # Split the images into class 0 and class 1
    class_0_images = train_images[train_labels == 0]
    class_1_images = train_images[train_labels == 1]
    class_0_labels = train_labels[train_labels == 0]
    class_1_labels = train_labels[train_labels == 1]
    
    # Number of samples to match class 0
    num_class_0 = len(class_0_images)
    num_class_1 = len(class_1_images)
    
    # If class 1 has fewer samples, oversample it
    if num_class_1 < num_class_0:
        # Randomly choose from class_1_images to oversample it
        oversample_indices = np.random.choice(np.arange(num_class_1), size=num_class_0 - num_class_1, replace=True)
        oversampled_class_1_images = class_1_images[oversample_indices]
        oversampled_class_1_labels = class_1_labels[oversample_indices]

        # Combine original class 1 images with the oversampled ones
        class_1_images = np.concatenate([class_1_images, oversampled_class_1_images], axis=0)
        class_1_labels = np.concatenate([class_1_labels, oversampled_class_1_labels], axis=0)

    # Concatenate class 0 and the new class 1 images to get the final balanced dataset
    train_images = np.concatenate([class_0_images, class_1_images], axis=0)
    train_labels = np.concatenate([class_0_labels, class_1_labels], axis=0)

    return train_images, val_images, test_images, train_labels, val_labels, test_labels

def get_isic2020_data_loaders(images, labels, train_bs=32, test_val_bs=320, aug_factor=1):
    """
    Returns: train_loader, val_loader, test_loader
    """
    # Preform the data split
    train_images, val_images, test_images, train_labels, val_labels, test_labels = train_val_test_split(images, labels)

    # Train dataset
    train_ds = TripletDataGenerator(
        images=train_images,
        labels=train_labels,
        train=True, 
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(degrees=10 * aug_factor, fill=(255, 255, 255)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1  * aug_factor, contrast=0.1 * aug_factor, saturation=0.1 * aug_factor, hue=0.05* aug_factor),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    )
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=4)

    # Validation dataset
    val_ds = TripletDataGenerator(
        images=val_images,
        labels=val_labels,
        train=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    )
    val_loader = DataLoader(val_ds, batch_size=test_val_bs, shuffle=True, num_workers=4)

    # Testing dataset
    test_ds = TripletDataGenerator(
        images=test_images,
        labels=test_labels,
        train=False,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )
    test_loader = DataLoader(test_ds, batch_size=test_val_bs, shuffle=True, num_workers=4)
    return train_loader, val_loader, test_loader