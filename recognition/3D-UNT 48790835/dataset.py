import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import nibabel as nib
import numpy as np
import torch

# Define 3D medical dataset class
class Medical3DDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transform=None):
        self.images_path = imgs_path
        self.labels_path = labels_path
        self.image_names = sorted(os.listdir(self.images_path))
        self.label_names = sorted(os.listdir(self.labels_path))
        self.transform = transform

        if len(self.image_names) != len(self.label_names):
            raise ValueError("The number of images and labels do not match. Please check the data.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label_name = self.label_names[idx]

        img_path = os.path.join(self.images_path, img_name)
        label_path = os.path.join(self.labels_path, label_name)

        # Load NIfTI images
        image = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.float32)

        # Expand dimensions to match (C, D, H, W) format
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # Convert to Tensor
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

# Define transforms for 3D images and labels if needed
def get_transform():
    return Compose([
        # Add 3D-specific transformations here if needed, like normalization, resizing, etc.
    ])
