"""
Dataset and dataloader classes for loading and pre-processing
the HIPMRI prostate cancer dataset.

@author George Reid-Smith
"""
import os

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class MRIDataset(Dataset):
    """
    Loads the directory of HipMRI prostate cancer nifti images into
    a PyTorch compatible dataset.
    """
    def __init__(self, directory: str, norm_image: bool=True, target_shape=(256, 128),
                 get_affines: bool=False, early_stop: bool=False, dtype = torch.float32):
        """Initialize dataset attributes.

        Args:
            directory (int): directory of the nifti files
            norm_image (bool, optional): normalize the image. Defaults to True.
            target_shape (tuple, optional): shape to resize input image. Defaults to (256, 128).
            get_affines (bool, optional): get affines. Defaults to False.
            early_stop (bool, optional): early stoppage for testing. Defaults to False.
            dtype (_type_, optional): target datatype. Defaults to torch.float32.
        """
        self.directory = directory
        self.image_names = [os.path.join(directory, fname) for fname in os.listdir(directory)]
        self.norm_image = norm_image
        self.get_affines = get_affines
        self.dtype = dtype
        self.resize = T.Resize(target_shape)
        self.normalize = T.Normalize((0.5), (0.5))
 
    def __len__(self):
        """Return length of the image dataset."""
        return len(self.image_names)

    def __getitem__(self, idx: int):
        """Returns a dataset item.

        Args:
            idx (int): index of item to be returned.

        Returns:
            any: dataset item. Default type of np.float32.
        """
        # Load image by index
        image_name = self.image_names[idx]
        nifti_image = nib.load(image_name)
        in_image = nifti_image.get_fdata(caching='unchanged')
        
        # Remove extra dims
        if len(in_image.shape) == 3:
            in_image = in_image[:,:,0] # Remove extra dims
        
        # Resize the image
        in_image = torch.tensor(in_image, dtype=self.dtype).unsqueeze(0)
        in_image = self.resize(in_image)
        
        # Normalize
        if self.norm_image:
            in_image = self.normalize(in_image)
        
        # Get image affines
        if self.get_affines:
            affine = in_image.affine
            return in_image, affine

        return in_image

class MRIDataloader:
    """Loads HipMRI prostate cancer images into a dataset and then 
    into a dataloader container.
    """
    def __init__(self, directory: str, batch_size: int, shuffle: bool, 
                 norm_image: bool=True, target_shape=(256, 128),
                 get_affines: bool=False, early_stop: bool=False, 
                 dtype = torch.float32):
        """Load the dataset and build a DataLoader container.

        Args:
            directory (int): directory of the nifti files
            batch_size (int): batch size for the loader
            shuffle (bool):  shuffle the images in the loader
            norm_image (bool, optional): normalize the image. Defaults to True.
            target_shape (tuple, optional): shape to resize input image. Defaults to (256, 128).
            get_affines (bool, optional): get affines. Defaults to False.
            early_stop (bool, optional): early stoppage for testing. Defaults to False.
            dtype (_type_, optional): target datatype. Defaults to torch.float32.
        """
        self.dataset = MRIDataset(directory, norm_image, target_shape, get_affines, early_stop, dtype)
        self.loader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle)

    def load(self):
        """Return the dataloader container of the dataset.

        Returns:
            DataLoader: the HIPMRI dataloader container
        """
        return self.loader
