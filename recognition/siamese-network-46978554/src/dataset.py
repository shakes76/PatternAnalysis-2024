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

    def __init__(
        self, mode="train", data_dir=DATA_DIR, transform=None, size=127, seed: int = 42
    ):
        """
        Args:
            mode: One of "train", "test", or "ref". Determines which split to use.
            data_dir: The directory in which the metadata CSVs can be found. The
              default directory structure (i.e. from the original archive file) is
              assumed.
            transform: A transform (or sequence of transforms) to apply to each image.
            size: Size of the reference dataset, preferably an odd number to avoid ties
              during classification. This value is ignored if mode != "ref".
            seed: A seed for reproducibility. This value is ignored if mode != "ref".
        """
        self.mode = mode
        self.data_dir = data_dir
        self.transform = transform
        if self.mode == "train":
            self.metadata = pd.read_csv(data_dir / "train-split-metadata.csv")
        elif self.mode == "test":
            self.metadata = pd.read_csv(data_dir / "test-split-metadata.csv")
        else:  # self.mode == "ref"
            metadata = pd.read_csv(data_dir / "train-split-metadata.csv")

            benign = metadata[metadata["target"] == 0]
            malign = metadata[metadata["target"] == 1]
            benign_sample = benign.sample(math.floor(size / 2), random_state=seed)
            malign_sample = malign.sample(math.ceil(size / 2), random_state=seed)

            self.metadata = pd.concat([benign_sample, malign_sample])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata.iloc[idx]["target"]
        img_name = self.metadata.iloc[idx]["isic_id"] + ".jpg"
        img_path = self.data_dir / f"train-image/image/{img_name}"

        # Normalise pixel values to [0, 1]
        img = read_image(img_path) / 255

        if self.transform:
            img = self.transform(img)

        return img, label
