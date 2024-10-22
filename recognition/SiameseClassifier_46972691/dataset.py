# dataset.py
# Loads the dataset, creates an ImageDataset class to handle the metadata of the images. Creates the Dataloaders and visualises a batch of data.
# Author: Harrison Martin

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


def visualise_class_distribution(df):
    """
    Visualise the distribution of classes in the dataset.
    """
    class_counts = df['target'].value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    plt.show()


def show_sample_images(df, image_folder, num_samples=5):
    """
    Display a few sample images from the dataset.
    """
    sample_df = df.sample(n=num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for idx, (i, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(image_folder, f"{row['isic_id']}.jpg")
        image = Image.open(img_path)
        axes[idx].imshow(image)
        axes[idx].axis('off')
        axes[idx].set_title(f"Label: {row['target']}")
    plt.show()


class ImageDataset(Dataset):
    def __init__(self, image_folder, df, transform=None):
        self.image_folder = image_folder
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_folder, f"{row['isic_id']}.jpg")
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row['target'], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, label



def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the DataFrame into training, validation, and test sets.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['target'], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['target'], random_state=random_state)
    return train_df, val_df, test_df

def visualise_batch(data_loader):
    """
    Visualize a batch of images and their labels.
    """
    batch = next(iter(data_loader))
    images, labels = batch
    images = images.numpy()
    
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    axes = axes.flatten()
    for idx in range(32):
        img = images[idx].transpose(1, 2, 0)
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
        img = img.clip(0, 1)

        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"Label: {labels[idx].item()}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    # Load metadata
    metadata_df = pd.read_csv('recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-metadata.csv')

    # Create splits from the data
    train_df, val_df, test_df = split_dataset(metadata_df)

    # Define basic transformations (can create a more complex one later with augmentation)
    basic_transforms = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure all images are the same size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create datasets
    image_path = 'recognition/SiameseClassifier_46972691/test_dataset_2020_Kaggle/train-image/image' # This is local on my computer, must be changed for Rangpur
    
    train_dataset = ImageDataset(image_folder=image_path, df=train_df, transform=basic_transforms)
    val_dataset = ImageDataset(image_folder=image_path, df=val_df, transform=basic_transforms)
    test_dataset = ImageDataset(image_folder=image_path, df=test_df, transform=basic_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=5)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=5)

    # Visualise a batch from the training loader
    visualise_batch(train_loader)