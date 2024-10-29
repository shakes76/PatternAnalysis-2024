# containing the data loader for loading and preprocessing your data

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Convert an array to a one-hot encoded channel format.

    Args:
        arr (np.ndarray): Input array to convert.
        dtype (np.dtype): Data type for the output array. Defaults to np.uint8.

    Returns:
        np.ndarray: One-hot encoded array.
    """
    channels = np.unique(arr)  # Get unique channel values
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)  # Initialize the result array
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1  # Set one-hot encoding
    return res

class MedicalImageDataset(Dataset):
    """Custom Dataset for loading medical images from NIfTI files."""

    def __init__(self, image_paths, norm_image=False, categorical=False, dtype=np.float32, early_stop=False):
        """
        Initialize the dataset.

        Args:
            image_paths (list): List of file paths to the images.
            norm_image (bool): Whether to normalize the images. Defaults to False.
            categorical (bool): Whether to convert images to categorical format. Defaults to False.
            dtype (np.dtype): Data type for the images. Defaults to np.float32.
            early_stop (bool): If True, stops loading early for quick testing. Defaults to False.
        """
        self.image_paths = image_paths  # Store image paths
        self.norm_image = norm_image  # Normalize flag
        self.categorical = categorical  # Categorical flag
        self.dtype = dtype  # Data type
        self.early_stop = early_stop  # Early stop flag
        self.images = self.load_data()  # Load and preprocess images

    def load_data(self):
        """Load and preprocess images from the given paths.

        Returns:
            np.ndarray: Preprocessed images.
        """
        num_images = len(self.image_paths)  # Total number of images
        first_case = nib.load(self.image_paths[0]).get_fdata(caching='unchanged')

        # Remove extra dimension if necessary
        if len(first_case.shape) == 3:
            first_case = first_case[:, :, 0]
        
        # Prepare the images array based on categorical flag
        if self.categorical:
            first_case = to_channels(first_case, dtype=self.dtype)  # Convert to channels
            rows, cols, channels = first_case.shape
            images = np.zeros((num_images, rows, cols, channels), dtype=self.dtype)
        else:
            rows, cols = first_case.shape
            images = np.zeros((num_images, rows, cols), dtype=self.dtype)

        # Load and preprocess each image
        for i, img_path in enumerate(tqdm(self.image_paths)):
            nifti_image = nib.load(img_path)  # Load NIfTI image
            image_data = nifti_image.get_fdata(caching='unchanged')  # Get image data

            if len(image_data.shape) == 3:
                image_data = image_data[:, :, 0]  # Remove extra dimension if necessary

            image_data = image_data.astype(self.dtype)  # Convert to specified dtype

            # Normalize the image if the flag is set
            if self.norm_image:
                image_data = (image_data - image_data.mean()) / image_data.std()

            # Convert to categorical format if the flag is set
            if self.categorical:
                image_data = to_channels(image_data, dtype=self.dtype)
                images[i, :, :, :] = image_data
            else:
                images[i, :, :] = image_data

            # Stop early if the early_stop flag is set
            if i > 20 and self.early_stop:
                break

        return images  # Return the preprocessed images

    def __len__(self):
        """Returns the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Fetch an image by index.

        Args:
            idx (int): Index of the image to fetch.

        Returns:
            torch.Tensor: The image tensor.
        """
        image = self.images[idx]  # Get the image at the specified index
        return torch.tensor(image, dtype=torch.float32)  # Return as a tensor


def get_dataloaders(train_dir, val_dir, test_dir, batch_size=8, num_workers=4, transform=None, norm_image=True):
    """
    Creates data loaders for training, validation, and testing datasets.

    Args:
        train_dir (str): Directory for training set.
        val_dir (str): Directory for validation set.
        test_dir (str): Directory for test set.
        batch_size (int): Size of each batch. Defaults to 8.
        num_workers (int): Number of worker threads for loading data. Defaults to 4.
        transform (callable, optional): Optional transform to be applied to each sample.
        norm_image (bool): Whether to normalize images. Defaults to True.

    Returns:
        tuple: Dataloaders for training, validation, and test sets.
    """
    # List of paths for each dataset
    train_images = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.nii.gz')]
    val_images = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.nii.gz')]
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith('.nii.gz')]

    # Create datasets for each set
    train_dataset = MedicalImageDataset(image_paths=train_images, norm_image=norm_image)
    val_dataset = MedicalImageDataset(image_paths=val_images, norm_image=norm_image)
    test_dataset = MedicalImageDataset(image_paths=test_images, norm_image=norm_image)

    # Create data loaders for each dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
