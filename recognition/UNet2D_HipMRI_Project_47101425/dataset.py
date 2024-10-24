import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from utils import load_data_2D


class MedicalImageDataset:
    def __init__(self, image_dir: str, mask_dir: str, normImage: bool = True, batch_size: int = 8, shuffle: bool = True):
        """
        Initialize the MedicalImageDataset class.

        Args:
            image_dir (str): Directory containing the medical image files.
            mask_dir (str): Directory containing the corresponding mask files.
            normImage (bool, optional): If True, normalize the images during loading. Defaults to True.
            batch_size (int, optional): Number of samples per batch. Defaults to 8.
            shuffle (bool, optional): If True, shuffle the dataset. Defaults to True.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normImage = normImage

        # Get image and mask files in a sorted order
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

        # Load image and mask data from given directory, ensuring categories and normalized images
        self.images = load_data_2D(self.image_files, normImage=self.normImage)
        self.masks = load_data_2D(self.mask_files, normImage=False, dtype=np.int8)

    def get_dataset(self) -> tf.data.Dataset:
        """
        Create and return a tf.data.Dataset object from loaded images and masks.

        Returns:
            tf.data.Dataset: A TensorFlow Dataset object containing image and mask pairs.
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks))

        # Shuffle dataset
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.images))
        dataset = dataset.batch(self.batch_size)

        return dataset
