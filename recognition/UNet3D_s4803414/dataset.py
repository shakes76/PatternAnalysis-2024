import torch
from torch.utils.data import Dataset
import nibabel as nib
import os
import numpy as np
import random
import scipy.ndimage

class MRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])
        self.transform = transform
        self.augment = augment
        self.target_size = (256, 256, 128)  # Fixed target size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and mask
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        # Normalize image
        image_min, image_max = image.min(), image.max()
        image = (image - image_min) / (image_max - image_min) if image_max != image_min else image

        # Convert to float32 for PyTorch
        image = image.astype(np.float32)

        # Convert mask to long type
        mask = mask.astype(np.int64)
        mask = np.clip(mask, 0, None)  # Ensure no negative values

        # Resize image and mask to fixed size (256, 256, 128)
        image = self.resize(image, self.target_size)
        mask = self.resize(mask, self.target_size, is_mask=True)

        # Data augmentation (randomly applied)
        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        # Add channel dimension to image and mask (for PyTorch 3D convs)
        image = np.expand_dims(image, axis=0)  # Shape: (1, 256, 256, 128)
        mask = np.expand_dims(mask, axis=0)  # Shape: (1, 256, 256, 128)

        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        return image_tensor, mask_tensor

    def resize(self, volume, target_size, is_mask=False):
        """Resize the image or mask to the target size (256, 256, 128)."""
        factors = (
            target_size[0] / volume.shape[0],
            target_size[1] / volume.shape[1],
            target_size[2] / volume.shape[2]
        )
        if is_mask:
            return scipy.ndimage.zoom(volume, factors, order=0)  # Nearest-neighbor for masks
        else:
            return scipy.ndimage.zoom(volume, factors, order=1)  # Bilinear interpolation for images

    def apply_augmentation(self, image, mask):
        """Applies random augmentations to the image and mask."""
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()  # Flip along depth
            mask = np.flip(mask, axis=2).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()  # Flip along width
            mask = np.flip(mask, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()  # Flip along height
            mask = np.flip(mask, axis=0).copy()

        if random.random() > 0.5:
            angle = random.uniform(-30, 30)  # Rotate between -30 and 30 degrees
            image = scipy.ndimage.rotate(image, angle, axes=(2, 3), reshape=False, mode='nearest')
            mask = scipy.ndimage.rotate(mask, angle, axes=(2, 3), reshape=False, mode='nearest', order=0)

        if random.random() > 0.5:
            zoom_factor = random.uniform(0.9, 1.1)  # Zoom between 90% and 110%
            image = scipy.ndimage.zoom(image, (1, zoom_factor, zoom_factor, zoom_factor), order=1)
            mask = scipy.ndimage.zoom(mask, (1, zoom_factor, zoom_factor, zoom_factor), order=0)

            # Resize back to target size after zoom
            image = self.resize(image, self.target_size)
            mask = self.resize(mask, self.target_size, is_mask=True)

        return image, mask


