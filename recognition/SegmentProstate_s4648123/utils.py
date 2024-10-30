import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Sequence, Mapping
from torch.utils.data._utils.collate import default_collate
from config import (IMAGE_DIR, MASK_DIR)


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Converts an array to one-hot encoded channels.

    Parameters:
    - arr: Input array with categorical values.
    - dtype: Data type for the output array.

    Returns:
    - One-hot encoded 4D NumPy array.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c: c + 1][arr == c] = 1
    return res


def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, get_affines=False,
                 early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.
    This function pre - allocates 5D arrays for conv3d to avoid excessive memory & usage .
    normImage : bool (normalise the image 0.0 -1.0)
    dtype: Type of the data. If dtype =np.uint8, it is assumed that the data is & labels 10
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick & loading and testing scripts .
    """
    affines = []
    # ~ interp = ' continuous '
    interp = 'linear '
    if dtype == np.uint8:  # assume labels
        interp = 'nearest '
    # get fixed size
    num = len(image_names)
    nifti_image = nib.load(image_names[0])
    first_case = nifti_image.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)
    for i, inName in enumerate(tqdm(image_names)):
        nifti_image = nib.load(inName)
        in_image = nifti_image.get_fdata(caching='unchanged')  # read disk only
        affine = nifti_image.affine
        if len(in_image.shape) == 4:
            in_image = in_image[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        in_image = in_image[:, :, :depth]  # clip slices
        in_image = in_image.astype(dtype)
        if norm_image:
            # ~ in_image= in_image/np.linalg.norm(in_image)
            # ~ in_image = 255.*in_image/in_image.max()
            in_image = (in_image - in_image.mean()) / in_image.std()
        if categorical:
            in_image = to_channels(in_image, dtype=dtype)
            # ~ images[i,:,:,:,:] = in_image
            images[i, :in_image.shape[0], :in_image.shape[1], : in_image.shape[2],
            :in_image.shape[3]] = in_image  # with pad
        else:
            # ~ images[i,:,:,:] = in_image
            images[i, :in_image.shape[0], :in_image.shape[1], : in_image.shape[2]] = in_image  # with pad
        affines.append(affine)
        if i > 20 and early_stop:
            break
    if get_affines:
        return images, affines
    else:
        return images


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


def compute_class_weights(target):
    """
    Compute class weights based on the inverse of class frequency.

    Args:
        target (Tensor): Ground truth labels (B, C, D, H, W).

    Returns:
        Tensor: Class weights of shape (num_classes,).
    """
    reduce_axis = [0] + list(range(2, target.dim()))  # Reduce over batch and spatial dims
    class_counts = torch.sum(target, dim=reduce_axis)  # (C,)
    total = torch.sum(class_counts)  # Total pixel count

    # Inverse frequency weighting
    class_weights = total / (class_counts + 1e-6)  # Avoid division by zero
    class_weights = class_weights / class_weights.sum()  # Normalize

    return class_weights


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