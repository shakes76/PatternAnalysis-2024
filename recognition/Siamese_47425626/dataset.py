import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import v2
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


LOCAL = True  # For my local machine
IMAGE_DIR = os.path.expanduser('~/Projects/COMP3710/siamese_project/dataset/train-image/image/') if not LOCAL else \
    os.path.expanduser('~/.kaggle/datasets/isic-2020-jpg-256x256-resized/train-image/image/')
ANOT_FILE = os.path.expanduser('~/Projects/COMP3710/siamese_project/dataset/train-metadata.csv') if not LOCAL else \
    os.path.expanduser('~/.kaggle/datasets/isic-2020-jpg-256x256-resized/train-metadata.csv')


class ISICKaggleDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # Create a dictionary for fast lookup of positive and negative samples
        self.label_to_indices = {label: self.img_labels[self.img_labels['target'] == label].index.tolist()
                                 for label in self.img_labels['target'].unique()}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Return a triplet (anchor, positive, negative).
        """
        # Anchor image and its label
        anchor_img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]) + '.jpg'
        anchor_image = read_image(anchor_img_path)
        anchor_label = self.img_labels.iloc[idx, 3]

        # Positive sample (same class as anchor)
        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img_path = os.path.join(self.img_dir, self.img_labels.iloc[positive_idx, 1]) + '.jpg'
        positive_image = read_image(positive_img_path)

        # Negative sample (different class from anchor)
        negative_label = random.choice(list(set(self.img_labels['target']) - {anchor_label}))
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img_path = os.path.join(self.img_dir, self.img_labels.iloc[negative_idx, 1]) + '.jpg'
        negative_image = read_image(negative_img_path)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


def count_class_instances(dataset, indices):
    """
    Count the number of instances of each class in the dataset for the given indices.
    """
    class_counts = {0: 0, 1: 0}

    for idx in indices:
        label = dataset.img_labels.iloc[idx, 3]  # The label is in the 4th column (index 3)
        class_counts[label] += 1

    return class_counts


def split_dataset(dataset, random_state=42):
    """
    Splits dataset into train, validation, and test sets with a balanced training set,
    and proportional 0's in validation and test sets. Applies data augmentation for training set.
    """
    # Extract the labels
    labels = dataset.img_labels.iloc[:, 3]

    # Get all indices of class 0 and class 1
    class_0_indices = labels[labels == 0].index.tolist()
    class_1_indices = labels[labels == 1].index.tolist()

    # Calculate split sizes based on 80/10/10
    num_class_1 = len(class_1_indices)
    train_size = int(0.8 * num_class_1)
    val_size = int(0.1 * num_class_1)
    test_size = num_class_1 - train_size - val_size  # Remaining 10%

    # Get indices for training, validation, and test sets based on class 1
    train_class_1_indices = class_1_indices[:train_size]
    val_class_1_indices = class_1_indices[train_size:train_size + val_size]
    test_class_1_indices = class_1_indices[train_size + val_size:]

    # For test and validation, we add proportional amounts of class 0
    proportion_class_0 = len(class_0_indices) / len(class_1_indices)
    val_class_0_indices = random.sample(class_0_indices, int(len(val_class_1_indices) * proportion_class_0))
    test_class_0_indices = random.sample(class_0_indices, int(len(test_class_1_indices) * proportion_class_0))

    # For training, use the same number of class 0 as class 1
    train_class_0_indices = random.sample(class_0_indices, len(train_class_1_indices))

    # Combine class 0 and class 1 for each set
    train_indices = train_class_0_indices + train_class_1_indices
    val_indices = val_class_0_indices + val_class_1_indices
    test_indices = test_class_0_indices + test_class_1_indices

    return train_indices, val_indices, test_indices


def augment_dataset(dataset, transform, num_copies=8):
    """
    Augments the dataset by applying the given transform to each image `num_copies` times.
    """
    datasets = [Subset(dataset, range(len(dataset)))]  # The original dataset
    for _ in range(num_copies - 1):  # Augment dataset
        augmented_dataset = Subset(dataset, range(len(dataset)))  # Create a copy of the original subset
        augmented_dataset.dataset.transform = transform  # Apply the transform
        datasets.append(augmented_dataset)
    return ConcatDataset(datasets)


def get_data_loaders(batch_size=32):
    """
    Prepares and returns the data loaders for training, validation, and test datasets.
    """
    # Initialise the affine transforms for training set (data augmentation)
    train_transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomAffine(
            degrees=(-10, 10),
            shear=[-0.3, 0.3],
            scale=(0.8, 1.2),
            translate=(0, 0.1),
        ),
        v2.ToDtype(torch.float32, scale=True)
    ])

    # No transforms for validation and test sets
    test_transform = None

    # Load the dataset with no transform initially (we'll apply transforms only to training set)
    dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, transform=test_transform)

    # Split the dataset into train, val, and test indices
    train_indices, val_indices, test_indices = split_dataset(dataset)

    # Create subsets for each split (no transforms for val/test, augment train)
    train_dataset = Subset(ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, transform=train_transform), train_indices)
    val_dataset = Subset(ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, transform=test_transform), val_indices)
    test_dataset = Subset(ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, transform=test_transform), test_indices)

    # Augment training set: 8x the number of samples
    augmented_train_dataset = augment_dataset(train_dataset, train_transform, num_copies=8)

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def show_image_grid(dataset, num_images=5):
    """
    Displays a grid of images from the dataset.
    """
    fig, ax = plt.subplots(3, num_images, figsize=(20, 12))

    for i in range(num_images):
        anchor_image, positive_image, negative_image = dataset[i]

        ax[0, i].imshow(anchor_image.permute(1, 2, 0))
        ax[0, i].set_title(f"Anchor {i}")
        ax[1, i].imshow(positive_image.permute(1, 2, 0))
        ax[1, i].set_title(f"Positive {i}")
        ax[2, i].imshow(negative_image.permute(1, 2, 0))
        ax[2, i].set_title(f"Negative {i}")

    plt.show()




