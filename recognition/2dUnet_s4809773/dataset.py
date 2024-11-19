"""
Author: Roman Kull
Description: 
    This code makes a dataloader for nifti image files, and resizes them all to a standard size of 256 x 128 (H x W)
"""

import os
import torch
import torch.utils.data as data
import nibabel as nib
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F  # Import for resizing tensors

class NiftiDataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, normImage=False, resize_to=(256, 128)):

        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.transform = transform
        self.normImage = normImage
        self.resize_to = resize_to  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and mask
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Normalize the image if required
        if self.normImage:
            image = (image - image.mean()) / image.std()

        # Add channel dimensions
        image = np.expand_dims(image, axis=0)
        mask = np.round(mask).astype(np.int64)  # Round and convert to integer for class indices
        mask = np.expand_dims(mask, axis=0)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)  # Long for Dice loss

         # Resize using torch.nn.functional.interpolate
        image = F.interpolate(image.unsqueeze(0), size=self.resize_to, mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=self.resize_to, mode='nearest').squeeze(0).long()

        # Apply any transforms as necessary
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask


def create_dataloaders(image_dir, mask_dir, batch_size, normImage=False):
    # Initialize the dataset
    dataset = NiftiDataset(image_dir, mask_dir, normImage=normImage)
    
    # Create data loaders
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader
