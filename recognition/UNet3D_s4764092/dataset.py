import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import torchio as tio
import torch
from torch.utils.data import Dataset, DataLoader, random_split

def load_data_3D(image_names, normalize=False, dtype=np.float32, target_shape=(96, 96, 96), augment=False):
    """
    Load 3D medical image data from specified file paths, with options for normalization and augmentation.

    Parameters:
    - image_names (list of str): List of file paths to Nifti files.
    - normalize (bool): If True, normalize the image to [0, 1] or zero mean and unit variance.
    - dtype (data-type): Data type of the output array, default is np.float32.
    - target_shape (tuple of int): Desired dimensions of the image, default is (96, 96, 96).
    - augment (bool): If True, apply data augmentation, default is False.

    Returns:
    - np.array: Array of loaded and processed image data.
    """
    num_images = len(image_names)
    images = np.zeros((num_images, *target_shape), dtype=dtype)
    augmenter = Augment(target_shape=target_shape) if augment else None

    for i, image_path in enumerate(tqdm(image_names, desc="Loading images")):
        nifti_image = nib.load(image_path)
        image_data = nifti_image.get_fdata()
        image_data = resize_image(image_data, target_shape)
        if augmenter:
            image_data = augmenter.apply_augmentation(image_data)
        if normalize:
            image_data = (image_data - np.mean(image_data)) / (np.std(image_data) + 1e-8)
        images[i] = image_data
    return images

def resize_image(image, target_shape):
    """
    Resize a 3D image to the desired dimensions using zoom interpolation.

    Parameters:
    - image (np.array): Original image data.
    - target_shape (tuple of int): Target dimensions for the image.

    Returns:
    - np.array: Resized image.
    """
    scale_factors = [n / o for n, o in zip(target_shape, image.shape)]
    return zoom(image, scale_factors, order=1)

class Augment:
    """
    Defines basic data augmentation operations, such as random flipping.
    """
    def __init__(self, target_shape=(96, 96, 96)):
        self.transform = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5)
        ])

    def apply_augmentation(self, image):
        """
        Apply the defined augmentations to a given image.

        Parameters:
        - image (np.array): Original image data.

        Returns:
        - np.array: Augmented image data.
        """
        image = tio.ScalarImage(tensor=torch.tensor(image).unsqueeze(0))
        image = self.transform(image)
        return image.data.numpy().squeeze()

class ProstateMRI3DDataset(Dataset):
    """
    Dataset class for handling prostate MRI data.
    """
    def __init__(self, mri_dir, labels_dir, target_shape=(96, 96, 96), augment=True):
        self.mri_files = sorted(os.listdir(mri_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        self.mri_dir = mri_dir
        self.labels_dir = labels_dir
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        mri_data = load_data_3D([mri_path], normalize=True, target_shape=self.target_shape, augment=self.augment)
        label_data = load_data_3D([label_path], target_shape=self.target_shape, dtype=np.uint8, augment=False)
        return torch.tensor(mri_data[0], dtype=torch.float32), torch.tensor(label_data[0], dtype=torch.long)

MRI_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
TARGET_SHAPE = (96, 96, 96)
BATCH_SIZE = 4

dataset = ProstateMRI3DDataset(MRI_DIR, LABEL_DIR, target_shape=TARGET_SHAPE, augment=True)
train_size = int(0.9 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
