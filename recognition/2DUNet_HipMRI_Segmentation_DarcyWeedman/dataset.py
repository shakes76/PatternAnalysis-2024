"""
dataset.py

Author: Darcy Weedman
Student ID: 45816985
COMP3710 HipMRI 2D UNet project
Semester 2, 2024
"""

import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from skimage.transform import resize

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Converts a label array to one-hot encoding across channels.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, 
                getAffines=False, early_stop=False, target_size=(256, 256)):
    """
    Load 2D Nifti images and masks, resizing them to target_size.
    
    Args:
        imageNames (list): List of file paths to Nifti images.
        normImage (bool): Whether to normalize images.
        categorical (bool): Whether masks are categorical (multi-class).
        dtype (np.dtype): Desired data type for images.
        getAffines (bool): Whether to return affine transformations.
        early_stop (bool): Whether to stop early after a certain number of images.
        target_size (tuple): Desired image size (rows, cols).
        
    Returns:
        numpy.ndarray: Array of images.
        list (optional): List of affine matrices if getAffines is True.
    """
    affines = []
    num = len(imageNames)
    logging.info(f"Expected image size: {target_size}")
    
    # Initialise based on whether categorical
    if categorical:
        # Assume masks have multiple channels after one-hot encoding
        example_image = nib.load(imageNames[0]).get_fdata(caching='unchanged')
        if len(example_image.shape) == 3:
            example_image = example_image[:, :, 0]  # Remove extra dimensions
        example_image = to_channels(example_image, dtype=dtype)
        resized_example = resize(example_image, target_size + (example_image.shape[-1],), 
                                 mode='constant', preserve_range=True)
        rows, cols, channels = resized_example.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        # Single channel (binary masks or grayscale images)
        example_image = nib.load(imageNames[0]).get_fdata(caching='unchanged')
        if len(example_image.shape) == 3:
            example_image = example_image[:, :, 0]  # Remove extra dimensions
        rows, cols = target_size
        images = np.zeros((num, rows, cols), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames, desc="Loading Images")):
        try:
            niftiImage = nib.load(inName)
            inImage = niftiImage.get_fdata(caching='unchanged').astype(dtype)
            affine = niftiImage.affine
            
            if len(inImage.shape) == 3:
                inImage = inImage[:, :, 0]  # Ensure 2D
            
            # Normalise image if required
            if normImage:
                if inImage.std() != 0:
                    inImage = (inImage - inImage.mean()) / inImage.std()
                else:
                    inImage = inImage - inImage.mean()
            
            # Resize image
            inImage_resized = resize(inImage, target_size, mode='constant', preserve_range=True)
            
            if categorical:
                inImage_resized = to_channels(inImage_resized, dtype=dtype)
                # Resize channels as well
                inImage_resized = resize(inImage_resized, target_size + (inImage_resized.shape[-1],), 
                                         mode='constant', preserve_range=True)
                images[i, :, :, :] = inImage_resized
            else:
                images[i, :, :] = inImage_resized
            
            affines.append(affine)
            
        except Exception as e:
            logging.error(f"Error loading image {inName}: {e}")
            if categorical:
                # Fill with zeros if categorical
                images[i, :, :, :] = 0
            else:
                # Fill with zeros if not categorical
                images[i, :, :] = 0
            continue  # Skip to next image
        
        if early_stop and i >= 20:
            break
    
    if getAffines:
        return images, affines
    else:
        return images

def load_nifti_image(file_path, normImage=False, dtype=np.float32):
    """
    Loads a single 2D Nifti image, applies normalization if specified.

    Args:
        file_path (str): Path to the Nifti file.
        normImage (bool): Whether to normalize the image.
        dtype (np.dtype): Desired data type for the image.

    Returns:
        np.ndarray: Loaded (and possibly normalized) image.
    """
    try:
        nifti_image = nib.load(file_path)
        image = nifti_image.get_fdata(caching='unchanged')
        if len(image.shape) == 3:
            image = image[:, :, 0]  # Remove extra dimensions if present
        image = image.astype(dtype)
        if normImage:
            image = (image - image.mean()) / image.std()
        return image
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None

class HipMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, norm=True, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.norm = norm
        self.target_size = target_size

        # List and sort image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

        # Ensure each image has a corresponding mask
        self.image_paths = []
        self.mask_paths = []
        for img_file in self.image_files:
            mask_file = img_file.replace('case_', 'seg_')
            if mask_file in self.mask_files:
                self.image_paths.append(os.path.join(image_dir, img_file))
                self.mask_paths.append(os.path.join(mask_dir, mask_file))
            else:
                logging.warning(f"No corresponding mask for image: {img_file}")

        logging.info(f"Number of valid image-mask pairs: {len(self.image_paths)}")

        # Load all data

        logging.info("Loading images...")
        self.images = load_data_2D(
            self.image_paths, 
            normImage=self.norm, 
            categorical=False, 
            target_size=self.target_size
        )
        logging.info("Loading masks...")
        self.masks = load_data_2D(
            self.mask_paths, 
            normImage=False, 
            categorical=False, 
            target_size=self.target_size
        )

        # Convert images to float32
        self.images = self.images.astype(np.float32)
        
        # Keep masks as is, but ensure they're integers
        self.masks = self.masks.astype(np.int64)

        self.num_classes = len(np.unique(self.masks))
        logging.info(f"Number of classes: {self.num_classes}")
        logging.info(f"Unique values in masks: {np.unique(self.masks)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.images[idx]  # Shape: (H, W)
        mask = self.masks[idx]    # Shape: (H, W)

        # Add channel dimension to image
        image = np.expand_dims(image, axis=0)  # (1, H, W)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)  # Use long for class indices

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def visualize_sample(self, idx):
        image, mask = self[idx]
        image = image.squeeze().numpy()
        mask = mask.squeeze().numpy()

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='nipy_spectral')
        plt.title('Multi-class Mask')
        plt.colorbar(label='Class')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("another_test.png")

if __name__ == "__main__":
    image_dir = "keras_slices_train"
    mask_dir = "keras_slices_seg_train"

    # Create the dataset
    dataset = HipMRIDataset(image_dir, mask_dir, norm=True, target_size=(256, 256))

    print(f"Dataset size: {len(dataset)}")

    # Visualise a few random samples to test this is working as intended
    num_samples = 3
    for _ in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        dataset.visualize_sample(idx)

    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Image shape: {dataset[0][0].shape}")
    print(f"Mask shape: {dataset[0][1].shape}")

    # Check unique values in masks
    unique_values = set()
    for i in range(len(dataset)):
        unique_values.update(np.unique(dataset[i][1].numpy()).tolist())
    print(f"Unique mask values: {sorted(unique_values)}")
