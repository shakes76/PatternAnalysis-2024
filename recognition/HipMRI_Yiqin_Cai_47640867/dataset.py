import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class ProstateDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None):
        self.image = nib.load(image_path).get_fdata()
        self.mask = nib.load(mask_path).get_fdata()
        self.transform = transform
        self.slices = self.image.shape[2]

    def __len__(self):
        return self.slices

    def __getitem__(self, idx):
        image_slice = self.image[:, :, idx]
        mask_slice = self.mask[:, :, idx]
        if self.transform:
            image_slice = self.transform(image_slice)
            mask_slice = self.transform(mask_slice)
        return torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0), torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0)
