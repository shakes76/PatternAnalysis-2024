"""
Custom PyTorch Dataset for Loading and Preprocessing NIfTI Files

This script defines the `NiftiDataset` class, a custom PyTorch `Dataset` 
designed to load medical imaging data in NIfTI format. It handles loading 
both image and label files, applies optional transformations, and ensures 
consistent input sizes by cropping the depth dimension when necessary.

Key Features:
- Loads image and label NIfTI files.
- Automatically crops depth dimensions to a fixed size of 128 slices 
    if the original depth exceeds this value.


@author Joseph Savage
"""

import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class NiftiDataset(Dataset):
    def __init__(
        self,
        image_filenames,
        label_filenames,
        transform=None,
        dtype=np.float32,
    ):
        self.image_filenames = list(image_filenames)
        self.label_filenames = list(label_filenames)
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Load image
        inName = self.image_filenames[idx]
        niftiImage = nib.load(inName)

        inImage = niftiImage.get_fdata(caching="unchanged")

        # Remove extra dimension if present
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  

        inImage = inImage.astype(self.dtype)

        # Crop depth if necessary (Some images are of shape [256, 256, 144] but most are [256, 256, 128])
        if inImage.shape[2] > 128:
            start_idx = (inImage.shape[2] - 128) // 2
            end_idx = start_idx + 128
            inImage = inImage[:, :, start_idx:end_idx]

        # if self.normImage:
        #     inImage = (inImage - inImage.mean()) / inImage.std()

        # Change to tensor and add channels dimension to image 
        inImage = torch.from_numpy(inImage)
        inImage = inImage.unsqueeze(0)  # Shape becomes (1, D, H, W)

        # Load label
        labelName = self.label_filenames[idx]
        labelImage = nib.load(labelName).get_fdata(caching="unchanged")

        # Remove extra dimension if present
        if len(labelImage.shape) == 4:
            labelImage = labelImage[:, :, :, 0]  

        # Integer for the label, as they are class indices 
        labelImage = labelImage.astype(np.int64)

        # Crop depth of label the same way we did for the inImage
        if labelImage.shape[2] > 128:
            start_idx = (labelImage.shape[2] - 128) // 2
            end_idx = start_idx + 128
            labelImage = labelImage[:, :, start_idx:end_idx]

        # Change to tensor and add channels dimension to image 
        labelImage = torch.from_numpy(labelImage)
        labelImage = labelImage.unsqueeze(0)  # Shape becomes (1, D, H, W)
        
        # Optionally apply transforms
        if self.transform:
            inImage = self.transform(inImage)
            labelImage = self.transform(labelImage)
        
        return inImage, labelImage

