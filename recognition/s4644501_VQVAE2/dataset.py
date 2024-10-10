import os

import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class MRIDataset(Dataset):
    """
    Loads the directory of HipMRI prostate cancer nifti images into
    a PyTorch compatible dataset.
    """
    def __init__(self, directory: str, norm_image: bool=True, 
                 get_affines: bool=False, early_stop: bool=False, dtype = np.float32):
        """Initialize the image dataset

        Args:
            directory (string): directory of the nifti files
            normImage (bool, optional): normalize the images to between 1.0 and 0.0. Defaults to True.
            getAffines (bool, optional): data contains affines. Defaults to False.
            early_stop (bool, optional): stops loading prematurely for testing. Defaults to False.
            dtype (any, optional): datatype to be converted to. Defaults to np.float32.
        """
        self.directory = directory
        self.image_names = [os.path.join(directory, fname) for fname in os.listdir(directory)]
        self.norm_image = norm_image
        self.get_affines = get_affines
        self.dtype = dtype
 
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
        
        # Update image datatype
        in_image = in_image.astype(self.dtype) # Update datatype

        # Normalize
        if self.norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()
        
        # Get image affines
        if self.get_affines:
            affine = in_image.affine
            return in_image, affine

        return in_image