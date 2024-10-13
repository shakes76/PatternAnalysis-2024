"""
Data loader for loading and preprocessing data

Before using the package code, a train test split should be created with
`create_train_test_split(DATA_DIR)`.
"""

import math
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.io import read_image
from util import DATA_DIR


def create_train_test_split(data_dir: Path, seed: int = 42, train_size: float = 0.8):
    """
    Creates a train test split from the train metadata, and stores the splits in
    data_dir/train-split-metadata.csv and data_dir/test-split-metadata.csv.

    Args:
        data_dir: The directory in which the original train-metadata.csv is stored.
          This is also the directory in which the resulting splits will be stored.
        seed: A seed for reproducibility.
        train_size: Size of the training split.
    """
    # Drop first column of indices
    metadata = pd.read_csv(data_dir / "train-metadata.csv").iloc[:, 1:]

    # Create a (stratified) train/test split from the train metadata, so each
    # split has an equal proportion of benign vs. malignant images
    train_metadata, test_metadata = train_test_split(
        metadata,
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=metadata["target"],
    )

    train_metadata = train_metadata.reset_index(drop=True)
    test_metadata = test_metadata.reset_index(drop=True)

    train_metadata.to_csv(data_dir / "train-split-metadata.csv", index=False)
    test_metadata.to_csv(data_dir / "test-split-metadata.csv", index=False)


class MelanomaSkinCancerDataset(Dataset):
    """Custom dataset of melanoma skin cancer image pairs"""

    def __init__(self, train, img_dir=DATA_DIR / "train-test-split", transform=None):
        self.train = train
        self.img_dir = img_dir
        self.transform = transform
        if self.train:
            self.metadata = pd.read_csv(img_dir / "train-pairs-metadata.csv")
        else:
            self.metadata = pd.read_csv(img_dir / "test-metadata.csv")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata.iloc[idx]["target"]
        subdir = "train" if self.train else "test"

        if self.train:
            img1_name = self.metadata.iloc[idx]["isic_id1"] + ".jpg"
            img2_name = self.metadata.iloc[idx]["isic_id2"] + ".jpg"
            img1_path = self.img_dir / f"{subdir}/{img1_name}"
            img2_path = self.img_dir / f"{subdir}/{img2_name}"

            image1 = read_image(img1_path) / 255
            image2 = read_image(img2_path) / 255

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            return image1, image2, label

        # Test dataset
        img_name = self.metadata.iloc[idx]["isic_id"] + ".jpg"
        img_path = self.img_dir / f"{subdir}/{img_name}"

        image = read_image(img_path) / 255

        if self.transform:
            image = self.transform(image)

        return image, label


class MelanomaSiameseReferenceDataset(Dataset):
    """
    Custom dataset of reference melanoma skin cancer images for Siamese network prediction
    """

    def __init__(self, img_dir=DATA_DIR / "train-test-split", size=8, seed: int = 42):
        self.img_dir = img_dir
        self.size = size

        metadata = pd.read_csv(img_dir / "train-metadata.csv")
        benign = metadata[metadata["target"] == 0]
        malign = metadata[metadata["target"] == 1]
        benign_sample = benign.sample(size // 2, random_state=seed)
        malign_sample = malign.sample(size // 2, random_state=seed)

        metadata = pd.concat([benign_sample, malign_sample])
        metadata = metadata.sample(frac=1, random_state=seed).reset_index(drop=True)
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata.iloc[idx]["target"]
        img_name = self.metadata.iloc[idx]["isic_id"] + ".jpg"
        img_path = self.img_dir / f"train/{img_name}"

        image = read_image(img_path) / 255

        return image, label
