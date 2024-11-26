import os
import pandas as pd
import random
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms


LOCAL = True  # For my local machine
IMAGE_DIR = os.path.expanduser('~/Projects/COMP3710/siamese_project/dataset/train-image/image/') if not LOCAL else \
    os.path.expanduser('~/.kaggle/datasets/isic-2020-jpg-256x256-resized/train-image/image/')
ANOT_FILE = os.path.expanduser('~/Projects/COMP3710/siamese_project/dataset/train-metadata.csv') if not LOCAL else \
    os.path.expanduser('~/.kaggle/datasets/isic-2020-jpg-256x256-resized/train-metadata.csv')


class ISICKaggleDataset(Dataset):
    def __init__(self, annotations_file, img_dir, indices=None, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        if indices is not None:
            # Filter labels using the provided indices
            self.img_labels = self.img_labels.iloc[indices].reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]) + '.jpg'
        image = read_image(img_path).float() / 255.0
        label = self.img_labels.iloc[idx, 3]

        if self.transform:
            image = self.transform(image)

        return image, label


# Function to split dataset
def split_dataset(metadata, test_size=0.2, val_size=0.1, random_state=42):
    labels = metadata['target']
    train_indices, test_indices = train_test_split(range(len(metadata)), test_size=test_size, stratify=labels, random_state=random_state)
    train_labels = labels.iloc[train_indices]
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size / (1 - test_size), stratify=train_labels,
                                                  random_state=random_state)
    return train_indices, val_indices, test_indices


def balance_dataset_indices(metadata, train_indices):
    """
    Balance the dataset by undersampling the majority class and oversampling the minority class.
    The goal is to have the total number of samples remain the same as the original train set size.
    """
    # Get the labels for the training set
    train_labels = metadata.iloc[train_indices]['target']

    # Get indices for each class
    positive_indices = train_labels[train_labels == 1].index.tolist()  # Class '1'
    negative_indices = train_labels[train_labels == 0].index.tolist()  # Class '0'

    # Determine the target number of samples per class
    target_per_class = len(train_indices) // 2  # Since we want balanced classes

    # Undersample Class '0' (majority)
    if len(negative_indices) > target_per_class:
        negative_indices = random.sample(negative_indices, target_per_class)

    # Oversample Class '1' (minority)
    if len(positive_indices) < target_per_class:
        positive_indices = random.choices(positive_indices, k=target_per_class)

    # Combine indices and shuffle
    balanced_indices = negative_indices + positive_indices
    random.shuffle(balanced_indices)

    return balanced_indices


# Modified get_data_loaders function
def get_data_loaders(batch_size=32):
    """
    Prepares and returns the data loaders for training, validation, and test datasets using a 70/20/10 split.
    Balances the training dataset using undersampling and oversampling.
    """
    # Load metadata from CSV
    metadata = pd.read_csv(ANOT_FILE)

    # Split the dataset into train, val, and test indices using stratified sampling
    train_indices, val_indices, test_indices = split_dataset(metadata)

    # Balance the training set using undersampling and oversampling
    balanced_train_indices = balance_dataset_indices(metadata, train_indices)

    # Define transformations for the training set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop(size=(224, 224)),
    ])

    # Create datasets for each split
    train_dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, indices=balanced_train_indices, transform=train_transform)
    val_dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, indices=val_indices)
    test_dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR, indices=test_indices)

    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=32)
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of batches in train loader: {len(train_loader)}")
    print(f"Number of batches in val loader: {len(val_loader)}")
    print(f"Number of batches in test loader: {len(test_loader)}")
    print(f"Classes in train loader: {train_loader.dataset.img_labels['target'].unique()}")
    print(f"Classes in val loader: {val_loader.dataset.img_labels['target'].unique()}")
    print(f"Classes in test loader: {test_loader.dataset.img_labels['target'].unique()}")
    print(f"Count of each class in train loader:\n{train_loader.dataset.img_labels['target'].value_counts()}")
    print(f"Count of each class in val loader:\n{val_loader.dataset.img_labels['target'].value_counts()}")
    print(f"Count of each class in test loader:\n{test_loader.dataset.img_labels['target'].value_counts()}")


    # Show sample images from the training set - 3 of each class
    import matplotlib.pyplot as plt

    # Define the classes
    classes = ['benign', 'malignant']

    # Create a figure
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    # Plot 3 samples for each class
    for i, cls in enumerate(classes):
        class_indices = train_loader.dataset.img_labels[train_loader.dataset.img_labels['target'] == i].index
        for j in range(3):
            img, label = train_loader.dataset[class_indices[j]]
            img = img.permute(1, 2, 0)  # Change the order of dimensions for plotting
            axs[i, j].imshow(img)
            axs[i, j].set_title(f"Class: {classes[label]}")
            axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png')

