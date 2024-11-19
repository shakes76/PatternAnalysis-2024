import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

# Define a Dataset class for loading 3D medical images
class Prostate3DDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None, normalize=True):
        """
        Args:
            data_dir (string): Directory with all the images.
            label_dir (string): Directory with all the labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            normalize (bool): Whether to normalize the image data.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
        self.normalize = normalize
        
        # Collect all image and label paths
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]
        self.label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii.gz') or f.endswith('.nii')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load the 3D medical image and label
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Optional normalization for the image
        if self.normalize:
            image = (image - np.mean(image)) / np.std(image)
        
        # Ensure the label is binary if it's for segmentation
        label = (label > 0).astype(np.float32)

        # Add channel dimension (1, depth, height, width)
        image = np.expand_dims(image, axis=0)  # Shape becomes (1, depth, height, width)
        label = np.expand_dims(label, axis=0)  # Shape becomes (1, depth, height, width)

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return torch.from_numpy(image).float(), torch.from_numpy(label).float()


# Load 3D data from files for batch processing
def load_data_3D(imageNames, labelNames, normImage=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Loads 3D data from files, with optional normalization and early stopping for testing.
    
    Args:
        imageNames (list): List of paths to image files.
        labelNames (list): List of paths to label files.
        normImage (bool): Whether to normalize the images.
        dtype (numpy.dtype): Data type for the arrays.
        getAffines (bool): Whether to return affine matrices.
        early_stop (bool): Stops loading after 20 images if True.
        
    Returns:
        tuple: Arrays of images and labels, and optionally affine matrices.
    """
    affines = []
    num = len(imageNames)

    first_case = nib.load(imageNames[0]).get_fdata()
    rows, cols, depth = first_case.shape
    images = np.zeros((num, 1, rows, cols, depth), dtype=dtype) 
    labels = np.zeros((num, 1, rows, cols, depth), dtype=dtype) 

    for i, (inName, labelName) in enumerate(tqdm(zip(imageNames, labelNames))):
        image = nib.load(inName).get_fdata().astype(dtype)
        label = nib.load(labelName).get_fdata().astype(dtype)

        # Normalize the image if specified
        if normImage:
            image = (image - image.mean()) / image.std()

        images[i, 0, :, :, :] = image  # Place in array with channel dimension
        labels[i, 0, :, :, :] = label  # Place in array with channel dimension

        # Stop early for testing if early_stop is enabled
        if early_stop and i >= 20:
            break

    if getAffines:
        return images, labels, affines
    else:
        return images, labels


# Helper function for handling labels (categorical)
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Converts a label array to a multi-channel format, where each unique label value becomes a separate channel.
    
    Args:
        arr (np.ndarray): Input label array.
        dtype (numpy.dtype): Desired output data type.
    
    Returns:
        np.ndarray: Multi-channel label array.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1  # Convert label to one-hot channel representation
    return res
