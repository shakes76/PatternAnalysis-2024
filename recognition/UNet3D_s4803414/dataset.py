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

