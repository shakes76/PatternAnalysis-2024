"""
dataset.py

Author: Alex Pitman
Student ID: 47443349
COMP3710 - HipMRI UNet2D Segmentation Project
Semester 2, 2024

Contains data pipeline functionality.
"""

import numpy as np
import nibabel as nib
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGES_MEAN, IMAGES_STD
from utils import TRAIN_IMG_DIR, TRAIN_MASK_DIR
from utils import set_seed, SEED
import random
from torchvision import transforms
import torch


def load_data_2D(imageNames, normImage=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load medical image data (NIfTI formate) from file names.

    Images are stored in a 3D Numpy array of shape (N, IMAGE_HEIGHT, IMAGE_WIDTH) where N is the number
    of images.

    normImage: bool (normalise the image 0.0 to 1.0 then distribute to zero mean and unit variance)
    early_stop: Stop loading pre-maturely which leaves array mostly empty. For quick loading and testing scripts.
    """
    num = len(imageNames)
    images = np.zeros((num, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=dtype)
    affines = []

    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # Read disk only

        affine = niftiImage.affine
        affines.append(affine)

        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # Sometimes extra dims in HipMRI study data -> remove
        inImage = inImage.astype(dtype)
        
        # Store image
        images[i,:,:] = inImage[:IMAGE_HEIGHT,:IMAGE_WIDTH] # Note: some images larger than 256x128 -> crop
        
        if i > 20 and early_stop:
            break

    if normImage:
        images = images / 255.0 # To [0.0, 1.0] scale 
        images = (images - IMAGES_MEAN) / IMAGES_STD # Distribute with zero mean and unit variance

    if getAffines:
        return images, affines
    else:
        return images

def get_names(data_path):
    """
    Gets image file names for input to load_data_2D
    """
    # Sorting alphabetically is very important
    names = sorted([os.path.join(data_path, img) for img in os.listdir(data_path) if img.endswith(('.nii', '.nii.gz'))])
    return names

class ProstateDataset(Dataset):
    """
    Stores images with their corresponding masks and performs necessary transformations.
    Used for DataLoader.
    """
    def __init__(self, image_dir, mask_dir, transforms=None, early_stop=False, normImage=False):
        self.imageNames = get_names(image_dir)
        self.maskNames = get_names(mask_dir)
        self.images = load_data_2D(self.imageNames, normImage=normImage, early_stop=early_stop)
        self.masks = load_data_2D(self.maskNames, normImage=False, early_stop=early_stop)
        self.transforms = transforms
        set_seed(SEED)

    def __len__(self):
		# Total number of samples contained in the dataset
        return len(self.imageNames)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        # Add channel dim
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert to tensor
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        if self.transforms is not None:
			# Apply the transformations to both image and its mask
            # Ensure same seed for both image and mask
            seed = random.randint(0, 2**32 - 1)
            
            set_seed(seed)
            image = self.transforms(image)
            set_seed(seed)
            mask = self.transforms(mask)
		
        return (image, mask)

def test():
    imageNames = get_names(TRAIN_IMG_DIR)
    images = load_data_2D(imageNames, normImage=True, early_stop=True)
    print(images.shape)
    plt.imshow(images[0])
    plt.show()

    maskNames = get_names(TRAIN_MASK_DIR)
    masks = load_data_2D(maskNames, normImage=False, early_stop=True)
    print(masks.shape)
    plt.imshow(masks[0])
    plt.show()

if __name__ == "__main__":
    test()