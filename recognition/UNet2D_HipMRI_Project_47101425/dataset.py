import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from utils import load_data_2D


class MedicalImageDataset:
    def __init__(self, image_dir, mask_dir, normImage=True, batch_size=8, shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normImage = normImage

        # Get image and mask files
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

        # Load images and masks
        self.images = load_data_2D(self.image_files, normImage=self.normImage)
        self.masks = load_data_2D(self.mask_files, normImage=False, dtype=np.int8)

    def get_dataset(self):
        """
        Create and return a tf.data.Dataset object from the loaded images and masks.
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.masks))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.images))
        dataset = dataset.batch(self.batch_size)

        return dataset
    
    def print_mask_categories(self):
        """
        Print the unique categories (label values) present in the mask dataset.
        """
        unique_labels = np.unique(self.masks)
        print("Unique mask categories (labels):", unique_labels)
