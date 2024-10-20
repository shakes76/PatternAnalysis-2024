import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import load_data_2D


class MedicalImageDataset:
    def __init__(self, image_dir, normImage=True, batch_size=8, shuffle=True):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normImage = normImage
        
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')]
        self.labels = np.zeros((len(self.image_files), 256, 144, 1), dtype=np.float32)

        self.images = load_data_2D(self.image_files, normImage=self.normImage)

    def get_dataset(self):
        """
        Create and return a tf.data.Dataset object from the loaded images and labels.
        """
        dataset = tf.data.Dataset.from_tensor_slices((self.images, self.labels))
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.images))
        dataset = dataset.batch(self.batch_size)
        
        return dataset
