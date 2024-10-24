import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class ProstateCancerDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image and mask files using nibabel
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = nib.load(image_path).get_fdata()  # Load the image data
        mask = nib.load(mask_path).get_fdata()    # Load the segmentation mask
        
        # Normalize the mask to be binary (0 or 1)
        mask = (mask > 0).astype(np.float32)  # Convert mask to binary (0, 1)

        # Ensure the data has the correct shape for PyTorch (Add channel dimension)
        image = np.expand_dims(image, axis=0)  # Shape: (1, H, W)
        mask = np.expand_dims(mask, axis=0)    # Shape: (1, H, W)

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        # Apply any optional transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
