import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
import matplotlib.pyplot as plt


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

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        The supplied CSV has 4 columns: id, imaage_name, patient_name, target (label)
        :param idx:
        :return:
        """
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1]) + '.jpg'
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def main():
    # Load the dataset
    dataset = ISICKaggleDataset(annotations_file=ANOT_FILE, img_dir=IMAGE_DIR)


if __name__ == '__main__':
    main()


