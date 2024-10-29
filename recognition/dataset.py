"""
Contains the data loaders for loading and preprocessing the ADNI brain dataset
This module will help prepare training, validation and testing data to be used 
"""

import torch
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as transforms
import random
import math
import copy

# ADNI data path
ADNI_PATH = Path('./AD_NC')

# Training image transforms
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Testing image transforms
TEST_TRANSFORM = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Rest of the original dataset.py content remains the same
class ADNIDataset(Dataset):
    """
    Custom Dataset class for the ADNI dataset.
    """
    def __init__(self, root_path, train=True, transform=None):
        self.path = Path(root_path, 'train' if train else 'test')
        self.transform = transform
        self.ad_files = os.listdir(Path(self.path, 'AD'))
        self.nc_files = os.listdir(Path(self.path, 'NC'))

    def __len__(self):
        return len(self.ad_files) + len(self.nc_files)

    def __getitem__(self, idx):
        assert idx >= 0 and idx < self.__len__(), "Index out of range."
        # Index in AD range
        if idx < len(self.ad_files):
            img_filename = self.ad_files[idx]
            image = Image.open(Path(self.path, 'AD', img_filename))
            label = 1
        # Index in NC range
        else:
            img_filename = self.nc_files[idx-len(self.ad_files)]
            image = Image.open(Path(self.path, 'NC', img_filename))
            label = 0
        # Apply transform if present
        if self.transform:
            image = self.transform(image)

        return image, label

def split_train_val(dataset: ADNIDataset, val_proportion: float, keep_proportion: float):
    """
    Function to split the training set into training and validation sets, with a certain proportion of images to be included in the validation set.

    Images to be split on a patient level (i.e. all images with the same patient ID to be included in the same set).
    """
    # Get the patient ID's and the number of AD and NC patients
    ad_patient_ids = set(filename.split('_')[0] for filename in dataset.ad_files)
    num_ad_patients = len(ad_patient_ids)
    nc_patient_ids = set(filename.split('_')[0] for filename in dataset.nc_files)
    num_nc_patients = len(nc_patient_ids)

    # Keep a certain proportion of the data. Used to cut training time for debugging purposes
    ad_patient_ids = random.sample(list(ad_patient_ids), math.floor(num_ad_patients*keep_proportion))
    num_ad_patients = len(ad_patient_ids)
    nc_patient_ids = random.sample(list(nc_patient_ids), math.floor(num_nc_patients*keep_proportion))
    num_nc_patients = len(nc_patient_ids)

    # Generate a random sample of patient ID's for the validation set
    ad_pids_val = random.sample(ad_patient_ids, math.floor(num_ad_patients*val_proportion))
    nc_pids_val = random.sample(nc_patient_ids, math.floor(num_nc_patients*val_proportion))

    # Make the validation dataset a deep copy of the training dataset
    val_dataset = copy.deepcopy(dataset)
    val_dataset.transform = TEST_TRANSFORM

    # Update each dataset's files by patient level split above
    dataset.ad_files = [file for file in dataset.ad_files if file.split('_')[0] not in ad_pids_val and file.split('_')[0] in ad_patient_ids]
    val_dataset.ad_files = [file for file in val_dataset.ad_files if file.split('_')[0] in ad_pids_val]
    dataset.nc_files = [file for file in dataset.nc_files if file.split('_')[0] not in nc_pids_val and file.split('_')[0] in nc_patient_ids]
    val_dataset.nc_files = [file for file in val_dataset.nc_files if file.split('_')[0] in nc_pids_val]

    return dataset, val_dataset


def get_dataloader(batch_size, train: bool, val_proportion: float = 0.2, keep_proportion: float = 1.0):
    """
    Returns data loader for either the training (plus validation) set, or the test set.
    """
    train_dataset = ADNIDataset(root_path=ADNI_PATH, train=True, transform=TRAIN_TRANSFORM)
    train_dataset, val_dataset = split_train_val(dataset=train_dataset, val_proportion=val_proportion, keep_proportion=keep_proportion)
    loader = (DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True))
    if train == False:
        dataset = ADNIDataset(root_path=ADNI_PATH, train=False, transform=TEST_TRANSFORM)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader