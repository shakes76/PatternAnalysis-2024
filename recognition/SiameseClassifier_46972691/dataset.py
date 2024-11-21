# dataset.py
# Handles datasets for the Siamese Network and Image Classifier.
# Author: Harrison Martin

import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split

# Class for the Siamese Network
class SiameseDataset(Dataset):
    def __init__(self, image_folder, df, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Separate images by class
        self.class0 = df[df['target'] == 0].reset_index(drop=True)
        self.class1 = df[df['target'] == 1].reset_index(drop=True)

        # Combine and reset index
        self.df = df.reset_index(drop=True)

    def __len__(self):
        # You can adjust this length if needed
        return len(self.df)

    def __getitem__(self, idx):
        # Randomly decide if this is a positive or negative pair
        if torch.rand(1).item() > 0.5:
            # Positive pair (same class)
            target = 1
            if torch.rand(1).item() > 0.5:
                # Class 0
                img_df = self.class0.sample(n=2, replace=True)
            else:
                # Class 1
                img_df = self.class1.sample(n=2, replace=True)
        else:
            # Negative pair (different classes)
            target = 0
            img1_df = self.class0.sample(n=1, replace=True)
            img2_df = self.class1.sample(n=1, replace=True)
            img_df = pd.concat([img1_df, img2_df])

        img_paths = [os.path.join(self.image_folder, f"{row['isic_id']}.jpg") for _, row in img_df.iterrows()]
        images = []
        for img_path in img_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Return image pair and target
        return images[0], images[1], torch.tensor([target], dtype=torch.float32)

# Class for the Image Classifier Network
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
        label = torch.tensor(row['target'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to split the dataset
def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the DataFrame into training, validation, and test sets.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['target'], random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, stratify=train_df['target'], random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)