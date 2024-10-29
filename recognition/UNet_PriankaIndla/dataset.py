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

    Attributes:
        seg_image_paths (list of str): Paths to the segmentation mask images.
        image_paths (list of str): Paths to the MRI images.
        normImage (bool): Whether to normalise images to a [0, 1] range.
        dtype (type): The desired data type of loaded images.
        target_shape (tuple): Target shape to resize images (height, width).
        transform (callable, optional): A function/transform to apply on the image and segmentation.
    """

    def __init__(self, seg_image_paths, image_paths, normImage=False, dtype=np.float32, target_shape=(256, 128), transform=None):
        """
        Initializes dataset by loading MRI images and segmentation masks and resizing them to a target shape.

        Args:
            seg_image_paths (list of str): List of paths to segmentation mask images.
            image_paths (list of str): List of paths to MRI images.
            normImage (bool): Whether to normalise images to a [0, 1] range.
            dtype (type): Data type for the loaded images.
            target_shape (tuple): Target shape for resizing the images.
            transform (callable, optional): Transformations to apply on the images and masks.
        """
        self.seg_image_paths = seg_image_paths
        self.image_paths = image_paths
        self.normImage = normImage
        self.dtype = dtype
        self.target_shape = target_shape
        self.transform = transform

        # Load images and segmentation masks
        self.images = self.load_data_2D(self.image_paths, normImage=self.normImage, dtype=self.dtype)
        self.seg_images = self.load_data_2D(self.seg_image_paths, normImage=self.normImage, dtype=self.dtype)

    def __getitem__(self, idx):
        """
        Retrieves image and segmentation mask by index, processes them, and returns them as tensors.

        Args:
            idx (int): Index of the image and segmentation mask to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The MRI image with shape [1, height, width].
                - seg (torch.Tensor): The segmentation mask with shape [1, height, width].
        """
        # Use pre-loaded images and segmentation masks
        image = self.images[idx]
        seg = self.seg_images[idx]

        # Image and segmentation are 2D, so add channel dimension to get shape [1, height, width]
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        if seg.ndim == 2:
            seg = np.expand_dims(seg, axis=0)

        # Normalize image to [0,1] if necessary
        if self.normImage:
            image = image / 255.0

        # Convert segmentation mask to binary (0 and 1 only)
        seg = (seg > 0).astype(np.float32)

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

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_paths)

    def get_image_dimensions(self):
        """
        Returns the dimensions of the first image in the dataset.

        Returns:
            tuple: The shape of the first image if the dataset is not empty, otherwise None.
        """
        if self.images.size > 0:
            return self.images[0].shape
        return None

    @staticmethod
    def load_data_2D(imageNames, normImage=False, dtype=np.float32, target_shape=(256, 128)):
        """
        Loads and processes a set of 2D images from the specified file paths, resizes them to a target shape,
        and normalises them if specified.

        Args:
            imageNames (list of str): Paths to image files.
            normImage (bool): Whether to normalise images to [0, 1].
            dtype (type): Desired data type for the loaded images.
            target_shape (tuple): Target shape for resizing the images.

        Returns:
            np.ndarray: An array containing processed images with the specified target shape and dtype.
        """
        num = len(imageNames)
        images = np.zeros((num, *target_shape), dtype=dtype)

        for i, inName in enumerate(tqdm(imageNames)):
            try:
                niftiImage = nib.load(inName)
                inImage = niftiImage.get_fdata(caching='unchanged').astype(dtype)

                if inImage.ndim == 2:
                    resized_image = resize(inImage, target_shape, mode='reflect', anti_aliasing=True)
                    if normImage:
                        resized_image = (resized_image - resized_image.mean()) / resized_image.std()
                    images[i] = resized_image
                    print(f"Loaded image {inName} with shape: {resized_image.shape}")
                else:
                    print(f"Warning: Expected 2D image but got shape {inImage.shape} for {inName}.")

            except Exception as e:
                print(f"Error loading image {inName}: {e}. Skipping this image.")

        print(f"Loaded {len(images)} images.")
        return images

