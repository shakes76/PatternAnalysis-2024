import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
import random

# Data generator class with data augmentation
class DataGenerator(keras.utils.Sequence):
    """
    This class manages the data when training the model.
    This is necessary since I have 8GB of memory and
    14GB of data.
    REF: this code is written with the help from chatGPT.
    """
    def __init__(self, image_files, label_files, batch_size=1, dim=(128, 128, 64), num_classes=6, shuffle=True):
        self.image_files = image_files
        self.label_files = label_files
        self.batch_size = batch_size
        self.dim = dim
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        """
        Calculates and returns the number of batches per epoch.
        """
        # ceil to account for the leftover data.
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index):
        """
        Generates and returns a single batch of data (images and labels) given the batch number.
        """
        # Generate indices of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Initialize
        X = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        # Generate data
        for i, idx in enumerate(indexes):
            # Load and preprocess image
            img = nib.load(self.image_files[idx]).get_fdata()
            lbl = nib.load(self.label_files[idx]).get_fdata()
            img = self.preprocess_image(img)
            lbl = self.preprocess_label(lbl)

            # Data augmentation
            img, lbl = self.augment(img, lbl)

            X[i, ..., 0] = img  # Add channel dimension
            y[i, ...] = lbl

        # One-hot encode labels
        y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        return X, y

    def on_epoch_end(self):
        """
        Prepares the generator for a new epoch by creating and optionally shuffling the indices of the data samples.
        """
        self.indexes = np.arange(len(self.image_files))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def preprocess_image(self, img):
        """
        process the data into the required format.
        """
        img = (img - np.mean(img)) / np.std(img)
        # Resize to target dimensions
        img = self.resize_volume(img)
        return img

    def preprocess_label(self, lbl):
        lbl = lbl.astype(np.uint8)
        lbl = self.resize_volume(lbl, interpolation='nearest')
        return lbl

    def resize_volume(self, volume, interpolation='linear'):
        # Resize across z-axis (depth)
        from scipy.ndimage import zoom
        depth_factor = self.dim[2] / volume.shape[2]
        width_factor = self.dim[0] / volume.shape[0]
        height_factor = self.dim[1] / volume.shape[1]
        if interpolation == 'linear':
            volume = zoom(volume, (width_factor, height_factor, depth_factor), order=1)
        elif interpolation == 'nearest':
            volume = zoom(volume, (width_factor, height_factor, depth_factor), order=0)
        return volume

    def augment(self, image, label):
        """
        This function flip or rotate the image at random.
        """
        # Random flip
        if random.random() > 0.5:
            axis = random.choice([0, 1])
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            image = ndimage.rotate(image, angle, axes=(0, 1), reshape=False, order=1, mode='nearest')
            label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False, order=0, mode='nearest')
        return image, label
