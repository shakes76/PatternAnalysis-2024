import kagglehub
import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ISICDataset(Dataset):
    def __init__(self, transform=None, undersample=False):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            undersample (bool): If True, undersample the majority class to match the minority class.
        """
        path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
        self.labels_df = pd.read_csv(os.path.join(path, "train-metadata.csv"))
        self.img_dir = os.path.join(path, "train-image/image")
        self.transform = transform

        # Precompute indices for both classes
        self.class_0_indices = self.labels_df[
            self.labels_df["target"] == 0
        ].index.tolist()
        self.class_1_indices = self.labels_df[
            self.labels_df["target"] == 1
        ].index.tolist()

        # If undersampling is enabled, match class 0 size to class 1 size
        if undersample:
            class_1_size = len(self.class_1_indices)
            self.class_0_indices = random.sample(self.class_0_indices, class_1_size)

        # Combine both class indices into one balanced index list
        self.balanced_indices = self.class_0_indices + self.class_1_indices
        random.shuffle(self.balanced_indices)

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        # First image and transformations
        real_idx = self.balanced_indices[idx]
        img1_name = self.labels_df.iloc[real_idx, 1] + ".jpg"
        img1_path = os.path.join(self.img_dir, img1_name)
        img1 = Image.open(img1_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)

        # Get the class of the current image
        img1_class = self.labels_df.iloc[real_idx, 3]

        # Randomly select the second image (either same or different class)
        if random.random() > 0.5:
            # Positive pair (same class)
            if img1_class == 0:
                img2_idx = random.choice(self.class_0_indices)
            else:
                img2_idx = random.choice(self.class_1_indices)
        else:
            # Negative pair (different class)
            if img1_class == 0:
                img2_idx = random.choice(self.class_1_indices)
            else:
                img2_idx = random.choice(self.class_0_indices)

        # Second image and transformations
        img2_name = self.labels_df.iloc[img2_idx, 1] + ".jpg"
        img2_path = os.path.join(self.img_dir, img2_name)
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img2 = self.transform(img2)

        # Label: 1 if same class, 0 if different class
        label = torch.tensor(
            (
                1
                if self.labels_df.iloc[real_idx, 3] == self.labels_df.iloc[img2_idx, 3]
                else 0
            ),
            dtype=torch.float32,
        )

        # Return the pair of images and the label
        return img1, img2, label


def getDataLoader(data_transforms, batch_size, undersample=False):
    isic_dataset = ISICDataset(data_transforms, undersample=undersample)
    dataloader = DataLoader(isic_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
