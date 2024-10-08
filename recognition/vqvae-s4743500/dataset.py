# dataset.py

import os 
import nibabel as nib # type: ignore
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms # type: ignore
from PIL import Image  # Import PIL for image handling

class ProstateMRIDataset(Dataset):
    def  __init__(self, img_dir, transform=None):
        """
        Args:
            img_dir (string): Path to the directory with MRI images.
            transform (callable, optional): Transformation to be applied on images
        """
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

        # Recursively finds all NIfTI files in the subdirectories
        self.image_paths = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    self.image_paths.append(os.path.join(root, file))
        
        self.image_paths.sort()  # Ensures consistent order

        # Adds a check to print a warning if no files were found
        if len(self.image_paths) == 0:
            print(f"No NIFTI files found in directory {img_dir}")

    def __len__(self):
        # Returns the number of images
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Loads the NIFTI image
        img_path = self.image_paths[idx]
        img_nifti = nib.load(img_path)
        img_data = img_nifti.get_fdata()

        # Selects middle slice if the image is 3D
        if img_data.ndim == 3:
            middle_slice = img_data.shape[2] // 2
            img_data = img_data[:, :, middle_slice]
        
        # Normalizes the image data to a range [0, 1] to help the model train better
        img_data_normalized = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        
        # Converts range to [0, 255] for image displaying
        img_data_255 = (img_data_normalized * 255).astype(np.uint8)

        # Converts the NumPy array to a PIL Image for transformations
        img = Image.fromarray(img_data_255)

        # # Comment out above and uncomment this for testing
        # img = Image.fromarray(np.uint8(img_data))

        # Applies transformations (such as resize, crop, normalization, etc.)
        if self.transform:
            img = self.transform(img)  

        return img