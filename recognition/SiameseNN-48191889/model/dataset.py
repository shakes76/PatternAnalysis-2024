import kagglehub
import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


class ISICDataset(Dataset):
    def __init__(self, transform=None, method="mixed", train=True, train_size=0.2):
        """
        Dataset of benign and malignant skin lesions from the ISIC 2020 Kaggle Challenge dataset.
        The data represents only the training data of the original dataset resized to 256 x 256
        taken from https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data.
        """

        # Download the dataset
        path = kagglehub.dataset_download("nischaydnk/isic-2020-jpg-256x256-resized")
        self.labels_df = pd.read_csv(os.path.join(path, "train-metadata.csv"))
        self.img_dir = os.path.join(path, "train-image/image")

        # Initialize transformations
        self.transform = transform

        # Initialize class indices
        self.c0_idx = self.labels_df[self.labels_df["target"] == 0].index.tolist()
        self.c1_idx = self.labels_df[self.labels_df["target"] == 1].index.tolist()

        print("Size of benign (majority/0) class: ", len(self.c0_idx))
        print("Size of malignant (minority/1) class: ", len(self.c1_idx))

        # Handle sampling methods
        if method == "undersampling":
            """
            Undersamples the majority class to match the size of the minority class.
            """

            if len(self.c0_idx) > len(self.c1_idx):
                self.c0_idx = random.sample(self.c0_idx, len(self.c1_idx))
            else:
                self.c1_idx = random.sample(self.c1_idx, len(self.c0_idx))

        elif method == "mixed":
            """
            Performs a mixture of undersampling and oversampling,
            oversamples the minority class while undersampling
            the majority class.
            """

            # Oversamples the minority class by a factor of 4
            # by duplicating the existing data. might switch to other methods if overfitting

            over_factor = 4
            self.c1_idx = self.c1_idx * over_factor

            # Undersamples the majority class by a factor of 0.5
            # by selecting random samples the length of the new size

            under_factor = 0.5
            self.c0_idx = random.sample(
                self.c0_idx, int(len(self.c0_idx) * under_factor)
            )

        print("\nResized benign class: ", len(self.c0_idx))
        print("Resized malignant class: ", len(self.c1_idx))
        print()

        # Combine class indices into one
        self.data_idx = self.c0_idx + self.c1_idx
        random.shuffle(self.data_idx)

        # Configure train condition
        train_batch = int(train_size * len(self.data_idx))

        if train:
            self.data_idx = self.data_idx[:train_batch]
            print("Train size:", len(self.data_idx))
        else:
            self.data_idx = self.data_idx[train_batch:]
            print("Test size:", len(self.data_idx))

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):

        # Fetch first image
        img1_idx = self.data_idx[idx]
        img1_name = self.labels_df.iloc[img1_idx, 1] + ".jpg"
        img1_path = os.path.join(self.img_dir, img1_name)

        img1 = Image.open(img1_path).convert("RGB")

        # Get image class
        img1_class = self.labels_df.iloc[img1_idx, 3]

        # Fetch second image to compare the first image to.
        # The second image will be randomly selected from the dataset with
        # a 50% class probability split
        if random.random() > 0.5:
            # Positive pair (same class)
            img2_idx = (
                random.choice(self.c0_idx)
                if img1_class == 0
                else random.choice(self.c1_idx)
            )
        else:
            # Negative pair (different class)
            img2_idx = (
                random.choice(self.c0_idx)
                if img1_class == 1
                else random.choice(self.c1_idx)
            )

        # Second image and transformations
        img2_name = self.labels_df.iloc[img2_idx, 1] + ".jpg"
        img2_path = os.path.join(self.img_dir, img2_name)

        img2 = Image.open(img2_path).convert("RGB")

        # Get image class
        img2_class = self.labels_df.iloc[img2_idx, 3]

        # Tranform image if applied
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # Return label based on image pair
        label = torch.tensor(
            (1 if img1_class == img2_class else 0), dtype=torch.float32
        )

        # Return the pair of images and the label
        return img1, img2, label


def getDataLoader(data_transforms, batch_size, method, train=True, train_size=0.8):
    isic_dataset = ISICDataset(
        data_transforms, method=method, train=train, train_size=train_size
    )
    dataloader = DataLoader(isic_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
