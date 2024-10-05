import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


class ISIC2020Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train', split_ratio=0.8):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode

        # Split data into train and test
        self.train_data, self.test_data = self.train_test_split(split_ratio)

        if mode == 'train':
            self.data = self.train_data
            # Augment minority class to match majority class
            benign_count = len(self.data[self.data['target'] == 0])
            malignant_samples = self.data[self.data['target'] == 1]
            # Augment factor is the number of times we need to repeat the malignant samples
            augment_factor = benign_count // len(malignant_samples) - 1
            augmented_malignant = pd.concat([malignant_samples] * augment_factor, ignore_index=True)
            self.data = pd.concat([self.data, augmented_malignant], ignore_index=True)
        else:
            self.data = self.test_data

        self.benign = self.data[self.data['target'] == 0]
        self.malignant = self.data[self.data['target'] == 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor_row = self.data.iloc[idx]
        anchor_img = self.get_image(anchor_row)
        anchor_label = anchor_row['target']

        if anchor_label == 0:  # benign
            positive = self.get_image(self.benign.sample().iloc[0])
            negative = self.get_image(self.malignant.sample().iloc[0])
        else:  # malignant
            positive = self.get_image(self.malignant.sample().iloc[0])
            negative = self.get_image(self.benign.sample().iloc[0])

        return anchor_img, positive, negative, anchor_label

