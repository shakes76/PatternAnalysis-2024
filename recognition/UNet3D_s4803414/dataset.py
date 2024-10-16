import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all the MRI volumes.
            mask_dir (str): Directory with all the segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load the MRI and mask volumes using nibabel
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        def crop_image(image, target_depth):
            # Crop to the target depth
            return image[:, :, :target_depth]

        if image.shape != mask.shape:
            mask = crop_image(mask, 128)
            image = crop_image(image, 128)

        # Normalize image (0 to 1 range)
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min) if image_max != image_min else image

        # Convert to float32 for PyTorch
        image = image.astype(np.float32)

        # Convert mask to long type for multi-class segmentation
        mask = mask.astype(np.int64)  # Convert to integer type
        mask = np.clip(mask, 0, None)  # Ensure no negative values

        # Add channel dimension to image and mask (for PyTorch 3D convs: [C, D, H, W])
        image = np.expand_dims(image, axis=0)  # Shape: (1, 256, 256, 128)
        mask = np.expand_dims(mask, axis=0)  # Shape: (1, 256, 256, 128)

        # Apply transforms if provided
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)  # Mask already in long format

        return image_tensor, mask_tensor
