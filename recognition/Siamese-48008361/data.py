import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class SiameseDataset(Dataset):
    def __init__(self, image_pairs, labels, image_dir, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img_name1, img_name2 = self.image_pairs[idx]
        label = self.labels[idx]

        # Load images
        img_path1 = os.path.join(self.image_dir, img_name1 + '.jpg')
        img_path2 = os.path.join(self.image_dir, img_name2 + '.jpg')
        image1 = Image.open(img_path1).convert('RGB')
        image2 = Image.open(img_path2).convert('RGB')

        # Apply transformations
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float32)
