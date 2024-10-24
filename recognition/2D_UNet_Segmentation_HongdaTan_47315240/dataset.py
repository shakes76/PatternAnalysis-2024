import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class ProstateCancerDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        image = np.expand_dims(image, axis=0)  # Add channel dimension
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    