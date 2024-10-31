import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from tqdm import tqdm
from niftiload import load_data_2D
import os

class NIFTIDataset(Dataset):
    def __init__(self, imageDir, normImage=False, categorical=False, dtype=np.float32, early_stop=False):
        """
        Custom Dataset for loading medical images and corresponding affines.
        
        Args:
        - imageNames (list): List of file paths to medical images.
        - normImage (bool): Whether to normalize the image.
        - categorical (bool): Whether to convert images to one-hot encoding.
        - dtype: Data type of the images.
        - early_stop (bool): Whether to stop loading prematurely (for testing).
        """
        self.imageNames = [os.path.join(imageDir, fname) for fname in os.listdir(imageDir) if fname.endswith('.nii.gz')]

        # Load images and affines using the load_data_2D function
        self.images, self.affines = load_data_2D(self.imageNames, normImage=normImage, 
                                                 categorical=categorical, dtype=dtype, 
                                                 getAffines=True, early_stop=early_stop)

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get image and affine at index `idx`.
        
        Args:
        - idx (int): Index of the image to fetch.
        
        Returns:
        - image (Tensor): Image at the specified index.
        - affine (ndarray): Affine matrix of the image.
        """
        image = self.images[idx]
        affine = self.affines[idx]
        
        # Convert the image to a PyTorch tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        
        return image_tensor, affine
