import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from typing import Sequence, Mapping
from torch.utils.data._utils.collate import default_collate
from config import (IMAGE_DIR, MASK_DIR)


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Converts an array to one-hot encoded channels with a fixed number of classes.

    Parameters:
    - arr: Input array with categorical values.
    - num_classes: Total number of classes to ensure consistent channel encoding.
    - dtype: Data type for the output array.

    Returns:
    - One-hot encoded 4D NumPy array with a fixed number of channels.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c: c + 1][arr == c] = 1
    return res


def load_image_and_label_3D(image_file, label_file, dtype=np.float32):
    """
    Load a 3D medical image and its corresponding label file.
    Parameters:
    - image_file: Path to the medical image (non-categorical).
    - label_file: Path to the label file (categorical).
    - dtype: Data type for the output arrays.

    Returns:
    - image: 4D NumPy array (1, rows, cols, depth) for the input image.
    - label: 4D NumPy array (channels, rows, cols, depth) for the categorical label.
    """

    # Load the image data (non-categorical)
    nifti_image = nib.load(image_file)
    image = nifti_image.get_fdata(caching='unchanged').astype(dtype)
    if len(image.shape) == 4:
        image = image[:, :, :, 0]  # Remove extra dimensions if present
    # Add a channel dimension at the front: (1, rows, cols, depth)
    image = np.expand_dims(image, axis=0)

    # Load the label data (categorical)
    nifti_label = nib.load(label_file)
    label = nifti_label.get_fdata(caching='unchanged').astype(np.uint8)
    if len(label.shape) == 4:
        label = label[:, :, :, 0]  # Remove extra dimensions if present
    # Convert label to categorical (one-hot encoded) format
    label = to_channels(label, dtype=dtype)
    # Reorder label to (channels, rows, cols, depth)
    label = np.transpose(label, (3, 0, 1, 2))

    return image, label


def get_images():
    def extract_keys(file_path):
        parts = os.path.basename(file_path).split('_')
        return parts[0], str(parts[1])[-1]

    # List of image and mask filepaths
    image_files = [os.path.join(IMAGE_DIR, fname) for fname in os.listdir(IMAGE_DIR) if fname.endswith('.nii.gz')]
    mask_files = [os.path.join(MASK_DIR, fname) for fname in os.listdir(MASK_DIR) if fname.endswith('.nii.gz')]
    image_files, mask_files = sorted(image_files, key=extract_keys), sorted(mask_files, key=extract_keys)

    return np.array(image_files), np.array(mask_files)


def collate_batch(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    collate_fn = default_collate
    if isinstance(elem, Mapping):
        batch_list = {}
        for k in elem:
            key = k
            data_for_batch = [d[key] for d in data]
            batch_list[key] = collate_fn(data_for_batch)
    else:
        batch_list = collate_fn(data)
    return batch_list


def plot_and_save(x, y_data, labels, title, xlabel, ylabel, filename):
    plt.figure()
    for y, label in zip(y_data, labels):
        plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()