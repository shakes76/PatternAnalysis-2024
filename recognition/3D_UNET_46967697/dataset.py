"""
This file contains the dataset class and data loader function for the 3D U-Net model. 
The dataset class loads the semantic MRs and labels from the specified paths. 
The data loader function splits the data into training and testing sets and returns the data loaders for training and testing. 
The data loader function is used in the training script to load the data for training the model.

@author Damian Bellew
"""

from utils import *

import numpy as np
import nibabel as nib
import torchio as tio
import tqdm 
import torch
import os

class Prostate3DDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading semantic MRs and their corresponding labels for 3D segmentation tasks.
    
    semantic_MRs_path (str): Path to the folder containing the MR images.
    semantic_labels_path (str): Path to the folder containing the segmentation labels.
    transforms (torchio.transforms): Optional transformations to apply to the dataset.
    """
    def __init__(self, semantic_MRs_path, semantic_labels_path, transforms=None):
        semantic_MRs_files = [f"{semantic_MRs_path}/{file}" for file in os.listdir(semantic_MRs_path)]
        semantic_labels_files = [f"{semantic_labels_path}/{file}" for file in os.listdir(semantic_labels_path)]

        semantic_MRs_files.sort()
        semantic_labels_files.sort()

        MRs = load_data_3D(semantic_MRs_files)
        labels = load_data_3D(semantic_labels_files)

        self.subjects = []

        # Create TorchIO subjects and apply optional transformations
        for mr, label in zip(MRs, labels):
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=torch.from_numpy(mr).unsqueeze(0).float()),
                mask=tio.LabelMap(tensor=torch.from_numpy(label).unsqueeze(0).long())
            )
            
            # Store transformed or original subjects
            if transforms:
                subject = transforms(subject)
            self.subjects.append(subject)

    def __len__(self):
        """
        Returns the number of subjects in the dataset.
        """
        return len(self.subjects)
    
    def __getitem__(self, idx):
        """
        Retrieves a subject by index.
        """
        subject = self.subjects[idx]

        # Extract image and mask tensors
        mr = subject['image'].data
        label = subject['mask'].data

        return mr, label
    

def get_data_loaders():
    """
    Creates and returns data loaders for training and testing using the Prostate3DDataset class. It applies data transformations.

    Returns:
        Tuple[DataLoader, DataLoader]: Data loaders for training and testing.
    """
    # Transforms
    transforms = tio.Compose([
        tio.ZNormalization(),
        tio.RescaleIntensity((0, 1)),
        tio.Resize((128,128,128)),
        tio.RandomFlip(),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5),
        tio.RandomElasticDeformation()
    ])

    # Data
    data = Prostate3DDataset(SEMANTIC_MRS_PATH, SEMANTIC_LABELS_PATH, transforms=transforms)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_data, test_data = torch.utils.data.random_split(data, [TRAIN_TEST_SPLIT, 1 - TRAIN_TEST_SPLIT], generator=generator)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data)
    test_loader = torch.utils.data.DataLoader(dataset=test_data)

    return train_loader, test_loader


def to_channels(arr: np.ndarray, dtype=np.uint8):
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c ] = 1
		
    return res

def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, get_affines=False, orient=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice 
        thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is 
        masks
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick 
    loading and testing scripts.
    """

    affines = []
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(image_names)
    nifti_image = nib.load(image_names[0])

    first_case = nifti_image.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm.tqdm(image_names)):
        nifti_image = nib.load(inName)

        in_image = nifti_image.get_fdata(caching='unchanged')  # read from disk only
        affine = nifti_image.affine

        if len(in_image.shape) == 4:
            in_image = in_image[:, :, :, 0]  # sometimes extra dims in HipMRI_study data

        in_image = in_image[:, :, :depth]  # clip slices
        in_image = in_image.astype(dtype)

        if norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()

        if categorical:
            in_image = to_channels(in_image, dtype=dtype)
            images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2], :in_image.shape[3]] = in_image  # with pad
        else:
            images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2]] = in_image  # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if get_affines:
        return images, affines
    else:
        return images