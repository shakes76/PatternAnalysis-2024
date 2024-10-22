"""
This file contains the data loader for preprocessing the data.

The data is augmented and then packed into loaders for the model.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
import skimage.transform


def resize_image(image, target_shape):
    """
    Resize a given image to the target shape using anti-aliasing.

    Args:
        image (np.ndarray): Input image.
        target_shape (tuple): The desired output shape (rows, cols).

    Returns:
        np.ndarray: Resized image.
    """
    return skimage.transform.resize(image, target_shape, mode = 'reflect', anti_aliasing = True)


def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    """
    Convert a grayscale image to a one-hot encoded image with multiple channels.

    Args:
        arr (np.ndarray): Input grayscale image.
        dtype (type, optional): The datatype for the output array. Defaults to np.uint8.

    Returns:
        np.ndarray: Image with a separate channel for each unique grayscale value.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype = dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res


def load_data_2D(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False, target_shape = (256, 128)):
    """
    Load and preprocess a list of 2D images from NIfTI files.

    Args:
        imageNames (list): List of paths to NIfTI files.
        normImage (bool, optional): Whether to normalize images by their mean and standard deviation. Defaults to False.
        categorical (bool, optional): Whether to convert images to one-hot encoded format. Defaults to False.
        dtype (type, optional): Data type for the images. Defaults to np.float32.
        getAffines (bool, optional): Whether to return affine matrices. Defaults to False.
        early_stop (bool, optional): If True, stop after processing 20 images. Defaults to False.
        target_shape (tuple, optional): Desired shape of the images. Defaults to (256, 128).

    Returns:
        np.ndarray or (np.ndarray, list): Preprocessed images, and optionally the affine matrices.
    """
    affines = []
    num = len(imageNames)
    
    # Load the first image to get the shape
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]

    # Use the first case's shape if no target shape is provided
    if target_shape is None:
        target_shape = first_case.shape

    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = target_shape[0], target_shape[1], first_case.shape[-1]
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = target_shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        # Resize the image to the target shape if needed
        if inImage.shape != target_shape:
            inImage = resize_image(inImage, target_shape)

        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


class VQVAENIfTIDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and processing NIfTI images for VQVAE.

    Attributes:
        data_dir (str): Path to the directory containing NIfTI files.
        transform (callable, optional): Optional transform to apply to each image.
        normImage (bool): Whether to normalize images by their mean and standard deviation.
        categorical (bool): Whether to convert images to one-hot encoded format.
        target_shape (tuple): Desired shape of the images.
        file_list (list): List of file paths for the NIfTI images.
    """
    def __init__(self, data_dir, transform = None, normImage = True, categorical = False, target_shape = (128, 128)):
        """
        Initialize the dataset with NIfTI images.

        Args:
            data_dir (str): Path to the directory containing NIfTI files.
            transform (callable, optional): Optional transform to apply to each image.
            normImage (bool, optional): Whether to normalize images by their mean and standard deviation. Defaults to True.
            categorical (bool, optional): Whether to convert images to one-hot encoded format. Defaults to False.
            target_shape (tuple, optional): Desired shape of the images. Defaults to (128, 128).
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normImage = normImage
        self.categorical = categorical
        self.target_shape = target_shape
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]


    def __len__(self):
        """
        Return the total number of images in the dataset.

        Returns:
            int: Number of images.
        """
        return len(self.file_list)


    def __getitem__(self, idx):
        """
        Load and return a preprocessed image at the given index.

        Args:
            idx (int): Index of the image.

        Returns:
            torch.Tensor: Preprocessed image as a PyTorch tensor.
        """
        # Lazy loading: load the NIfTI file when accessing this index
        inName = self.file_list[idx]
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged').astype(np.float32)

        # Handle 3D to 2D slice extraction
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        if self.normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        # Resize image to target shape (256, 128)
        if inImage.shape != self.target_shape:
            inImage = resize_image(inImage, self.target_shape)

        # Convert to categorical if needed
        if self.categorical:
            inImage = to_channels(inImage, dtype=np.float32)

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(inImage).float()

        # Add channel dimension if it's a 2D image
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor.permute(1, 2, 0)) # (H, W, C) for transform
            image_tensor = image_tensor.permute(0, 1, 2) # Back to (C, H, W)

        return image_tensor


def create_nifti_data_loaders(data_dir, batch_size, num_workers = 4, normImage = True, categorical = False, target_shape = (256, 128)):
    """
    Create PyTorch data loaders for NIfTI image datasets.

    Args:
        data_dir (str): Directory containing NIfTI files.
        batch_size (int): Number of images per batch.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        normImage (bool, optional): Whether to normalize images by their mean and standard deviation. Defaults to True.
        categorical (bool, optional): Whether to convert images to one-hot encoded format. Defaults to False.
        target_shape (tuple, optional): Desired shape of the images. Defaults to (256, 128).

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    dataset = VQVAENIfTIDataset(data_dir, normImage = normImage, categorical = categorical, target_shape = target_shape)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

    return data_loader
