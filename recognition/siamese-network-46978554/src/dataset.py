"""Data loader for loading and preprocessing data"""

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from util import DATA_DIR


def create_train_test_split(data_dir: Path, seed: int = 42, train_split: float = 0.8):
    import os
    import shutil

    rng = np.random.default_rng(seed)

    # Create train/test split in subdirectory so original data isn't affected
    out_dir = data_dir / "train-test-split"

    os.makedirs(out_dir / "train", exist_ok=True)
    os.makedirs(out_dir / "test", exist_ok=True)

    metadata = pd.read_csv(data_dir / "train-metadata.csv")

    # Shuffle data and create train/test split
    idxs = np.arange(len(metadata))
    rng.shuffle(idxs)
    cutoff = int(len(metadata) * train_split)
    train_idxs = idxs[:cutoff]
    test_idxs = idxs[cutoff:]

    def create_split(idxs: np.ndarray, split_name: str):
        md_split = metadata.iloc[idxs]

        if split_name == "train":
            md_split_benign = md_split[md_split["target"] == 0]
            md_split_malign = md_split[md_split["target"] == 1]

            md = pd.concat(
                [
                    _create_positive_pairs(md_split_benign, seed),
                    _create_positive_pairs(md_split_malign, seed),
                    _create_negative_pairs(md_split_malign, md_split_benign, seed),
                ]
            )

            md = md.sample(frac=1, random_state=seed).reset_index(drop=True)
            md.to_csv(out_dir / f"{split_name}-metadata.csv", index=False)
        else:
            md_split.to_csv(out_dir / f"{split_name}-metadata.csv", index=False)

        for isic_id in md_split["isic_id"]:
            src = data_dir / f"train-image/image/{isic_id}.jpg"
            dest = out_dir / f"{split_name}/{isic_id}.jpg"
            shutil.move(src, dest)

    create_split(train_idxs, "train")
    create_split(test_idxs, "test")


def _create_positive_pairs(md: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)

    idxs = np.arange(len(md))
    rng.shuffle(idxs)
    cutoff = len(idxs) // 2

    pairs = dict()
    pairs["isic_id1"] = md["isic_id"].iloc[:cutoff].reset_index(drop=True)
    pairs["isic_id2"] = md["isic_id"].iloc[cutoff:].reset_index(drop=True)
    pairs["target"] = [0] * (len(idxs) // 2)  # positive pairs ==> label is 0
    pairs = pd.DataFrame.from_dict(pairs)

    return pairs


def _create_negative_pairs(md1: pd.DataFrame, md2: pd.DataFrame, seed: int = 42):
    rng = np.random.default_rng(seed)

    # Match half of data in the larger class to smaller class
    # md1 assumed to be smaller class
    md2_idxs = np.arange(len(md2) // 2)
    md1_idxs = rng.choice(np.arange(len(md1)), len(md2_idxs))

    pairs = dict()
    pairs["isic_id1"] = md1["isic_id"].iloc[md1_idxs].reset_index(drop=True)
    pairs["isic_id2"] = md2["isic_id"].iloc[md2_idxs].reset_index(drop=True)
    pairs["target"] = [1] * (len(md2) // 2)  # negative pairs ==> label is 1
    pairs = pd.DataFrame.from_dict(pairs)

    return pairs


class MelanomaSkinCancerDataset(Dataset):
    """Custom dataset of melanoma skin cancer image pairs"""

    def __init__(self, train, img_dir=DATA_DIR / "train-test-split", transform=None):
        self.train = train
        self.img_dir = img_dir
        self.transform = transform
        if self.train:
            self.metadata = pd.read_csv(img_dir / "train-metadata.csv")
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

    def __init__(self, img_dir=DATA_DIR / "train-test-split", size=256, seed: int = 42):
        self.img_dir = img_dir
        self.size = size

        metadata = pd.read_csv(img_dir.parent / "train-metadata.csv")
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
