"""
This file contains the data loader for preprocessing the data.

The data is augmented and then packed into loaders for the model.
"""

"""
after the images are converted to numpy arrays and should i convert the images to tensors and then
load them into datasets?
    based on the batch size of the loader, only take the required images and turn them into arrays
    and tensors for each epoch
"""
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import os
from tqdm import tqdm
from torchvision import transforms
import skimage.transform


transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.RandomCrop(64),
    transforms.RandomHorizontalFlip(),
])


def resize_image(image, target_shape):
    return skimage.transform.resize(image, target_shape, mode = 'reflect', anti_aliasing = True)


def to_channels(arr: np.ndarray, dtype = np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype = dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res


def load_data_2D(imageNames, normImage = False, categorical = False, dtype = np.float32, getAffines = False, early_stop = False, target_shape = None):
    affines = []
    num = len(imageNames)
    
    # Load the first image to get the shape
    first_case = nib.load(imageNames[0]).get_fdata(caching = 'unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]

    # Use the first case's shape if no target shape is provided
    if target_shape is None:
        target_shape = first_case.shape

    if categorical:
        first_case = to_channels(first_case, dtype = dtype)
        rows, cols, channels = target_shape[0], target_shape[1], first_case.shape[-1]
        images = np.zeros((num, rows, cols, channels), dtype = dtype)
    else:
        rows, cols = target_shape
        images = np.zeros((num, rows, cols), dtype = dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching = 'unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        # Resize the image to the target shape if needed
        if inImage.shape != target_shape:
            inImage = resize_image(inImage, target_shape)

        if categorical:
            inImage = to_channels(inImage, dtype = dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


class VQVAENIfTIDataset(Dataset):
    def __init__(self, data_dir, transform = transform, normImage = True, categorical = False, target_shape = (128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.normImage = normImage
        self.categorical = categorical
        self.target_shape = target_shape
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
        # self.images = load_data_2D(self.file_list, normImage = self.normImage, categorical = self.categorical)


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx):
        # image = self.images[idx]

        # Lazy loading: load the NIfTI file when accessing this index
        inName = self.file_list[idx]
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged').astype(np.float32)

        # Handle 3D to 2D slice extraction
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]

        if self.normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        # Resize image to target shape (256, 128)
        if inImage.shape != self.target_shape:
            inImage = resize_image(inImage, self.target_shape)

        # Convert to categorical if needed
        if self.categorical:
            inImage = to_channels(inImage, dtype=np.float32)

        # Convert to PyTorch tensor
        image_tensor = torch.from_numpy(inImage).float()

        # Add channel dimension if it's a 2D image
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor.permute(1, 2, 0))  # (H, W, C) for transform
            image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor


def create_nifti_data_loaders(data_dir, batch_size, num_workers = 4, normImage = True, categorical = False, target_shape = (256, 128)):
    dataset = VQVAENIfTIDataset(data_dir, normImage = normImage, categorical = categorical, target_shape = target_shape)
    data_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    return data_loader
