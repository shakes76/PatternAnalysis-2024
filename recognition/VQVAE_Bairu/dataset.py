"""
Custom dataset loader for processing medical images from NIfTI files.

Author: Bairu An, s4702833.
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import torchvision.transforms as transforms

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a 2D array to a one-hot encoded channel format.
    
    Args:
        arr (np.ndarray): Input array containing categorical data.
        dtype (np.dtype): Data type for the output array.

    Returns:
        np.ndarray: One-hot encoded array with channels corresponding to unique values in the input.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res

class MedicalImageDataset(Dataset):
    """
    Custom Dataset for loading medical images from NIfTI files.

    Args:
        image_paths (list): List of file paths to the NIfTI images.
        norm_image (bool): Whether to normalize the images.
        categorical (bool): Whether to convert images to categorical format.
        dtype (np.dtype): Data type for the images.
        early_stop (bool): If True, stops loading after a certain number of images.
    """

    def __init__(self, image_paths, norm_image=False, categorical=False, dtype=np.float32, early_stop=False):
        self.image_paths = image_paths  # Store paths to the images
        self.norm_image = norm_image      # Flag for normalization
        self.categorical = categorical     # Flag for categorical data conversion
        self.dtype = dtype                 # Data type for images
        self.early_stop = early_stop       # Flag for early stopping
        self.images = self.load_data()     # Load and preprocess the data

    def load_data(self):
        """Load and preprocess images from the specified paths."""
        num_images = len(self.image_paths)
        first_case = nib.load(self.image_paths[0]).get_fdata(caching='unchanged')

        if len(first_case.shape) == 3:
            first_case = first_case[:, :, 0]  # Remove extra dimension if necessary

        # Prepare the images array based on whether categorical encoding is required
        if self.categorical:
            first_case = to_channels(first_case, dtype=self.dtype)
            rows, cols, channels = first_case.shape
            images = np.zeros((num_images, rows, cols, channels), dtype=self.dtype)
        else:
            rows, cols = first_case.shape
            images = np.zeros((num_images, rows, cols), dtype=self.dtype)

        # Load and preprocess each image
        for i, img_path in enumerate(tqdm(self.image_paths)):
            nifti_image = nib.load(img_path)
            image_data = nifti_image.get_fdata(caching='unchanged')

            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]  # Select the first channel if there are multiple

            image_data = image_data.astype(self.dtype)  # Convert to specified dtype

            # Resize image to (256, 128)
            image_data = self.resize_image(image_data)

            if self.norm_image:
                image_data = (image_data - image_data.mean()) / image_data.std()

            if self.categorical:
                image_data = to_channels(image_data, dtype=self.dtype)
                images[i, :, :, :] = image_data  # Store one-hot encoded image
            else:
                images[i, :, :] = image_data  # Store normal image

            if i > 20 and self.early_stop:
                break  # Stop loading images if early stop is enabled

        return images  # Return the preprocessed images

    def resize_image(self, image_data):
        """
        Resize the image to the required dimension of 256x128.

        Args:
            image_data (np.ndarray): The image data to be resized.

        Returns:
            np.ndarray: Resized image data.
        """
        # Check the current shape and resize accordingly
        if image_data.shape[0] != 256 or image_data.shape[1] != 128:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((256, 128)),  # Resize to (256, 128)
            ])
            image_data = transform(image_data)  
            return image_data.numpy()

        return image_data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get the image at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            torch.Tensor: The image tensor with an additional channel dimension.
        """
        image = self.images[idx]  
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension


def get_dataloaders(train_dir, val_dir, test_dir, batch_size=8, num_workers=4, transform=None, norm_image=True):
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        train_dir (str): Directory containing training images.
        val_dir (str): Directory containing validation images.
        test_dir (str): Directory containing testing images.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        transform (callable, optional): Optional transform to be applied to the images.
        norm_image (bool): Whether to normalize the images.

    Returns:
        tuple: Data loaders for training, validation, and test datasets.
    """
    train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.nii.gz')]
    val_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.nii.gz')]
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.nii.gz')]

    # Create dataset instances
    train_dataset = MedicalImageDataset(image_paths=train_images, norm_image=norm_image)
    val_dataset = MedicalImageDataset(image_paths=val_images, norm_image=norm_image)
    test_dataset = MedicalImageDataset(image_paths=test_images, norm_image=norm_image)

    # Create data loaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
