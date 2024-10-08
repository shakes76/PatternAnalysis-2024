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