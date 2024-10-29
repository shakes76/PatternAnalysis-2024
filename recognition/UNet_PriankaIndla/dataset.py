import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm
import torchvision.transforms as T  # Import for transformations

class ProstateMRIDataset(Dataset):
    """
    Dataset class for loading prostate MRI images and corresponding segmentation masks.
    The dataset loads 2D images, resizes them to a target shape, and normalizes them if necessary.
    """

    def __init__(self, seg_image_paths, image_paths, normImage=False, dtype=np.float32, target_shape=(256, 128), transform=None):
        """
        Initializes the dataset by loading MRI images and segmentation masks and resizing them to a target shape.
        """
        self.seg_image_paths = seg_image_paths
        self.image_paths = image_paths
        self.normImage = normImage
        self.dtype = dtype
        self.target_shape = target_shape  # Set the target shape for resizing
        self.transform = transform  # Optional transform

        # Load images and segmentation masks
        self.images = self.load_data_2D(self.image_paths, normImage=self.normImage, dtype=self.dtype)
        self.seg_images = self.load_data_2D(self.seg_image_paths, normImage=self.normImage, dtype=self.dtype)

    def __getitem__(self, idx):
        """
        Retrieves image and segmentation mask by index, processes them, and returns them as tensors.
        """
        # Use pre-loaded images and segmentation masks
        image = self.images[idx]  # Access pre-loaded image
        seg = self.seg_images[idx]  # Access pre-loaded segmentation mask

        # Image is 2D (grayscale), so add channel dimension to get shape [1, height, width]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        # Segmentation is 2D as well so add channel dimension
        if seg.ndim == 2:
            seg = np.expand_dims(seg, axis=0)

        # Normalize image to [0,1] if necessary
        if self.normImage:
            image = image / 255.0

        # Convert segmentation mask to binary (0 and 1 only)
        seg = (seg > 0).astype(np.float32)  # Set all values > 0 to 1, otherwise 0

        # Convert image and segmentation to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        seg = torch.tensor(seg, dtype=torch.float32)

        # Apply transformations if provided
        if self.transform:
            combined = torch.cat([image, seg], dim=0)  # Concatenate for consistent transform
            combined = self.transform(combined)
            image, seg = combined[0:1, :, :], combined[1:, :, :]

        return image, seg

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)

    def get_image_dimensions(self):
        """
        Returns the dimensions of the first image in the dataset.
        """
        if self.images.size > 0:
            return self.images[0].shape
        return None

    @staticmethod
    def load_data_2D(imageNames, normImage=False, dtype=np.float32, target_shape=(256, 128)):
        """
        Loads and processes a set of 2D images from the specified file paths, resizes them to a target shape,
        and normalizes them if specified.
        """
        num = len(imageNames)
        images = np.zeros((num, *target_shape), dtype=dtype)  # Initialize with target shape

        for i, inName in enumerate(tqdm(imageNames)):
            try:
                niftiImage = nib.load(inName)
                inImage = niftiImage.get_fdata(caching='unchanged').astype(dtype)

                if len(inImage.shape) == 2:
                    # Resize the single 2D slice to target shape
                    resized_image = resize(inImage, target_shape, mode='reflect', anti_aliasing=True)

                    if normImage:
                        resized_image = (resized_image - resized_image.mean()) / resized_image.std()

                    images[i] = resized_image  # Assign the resized image to the array
                    print(f"Loaded image {inName} with shape: {resized_image.shape}")  # Print dimensions for debugging
                else:
                    print(f"Warning: Expected 2D image but got shape {inImage.shape} for {inName}.")

            except Exception as e:
                print(f"Error loading image {inName}: {e}. Skipping this image.")

        print(f"Loaded {len(images)} images.")
        return images

