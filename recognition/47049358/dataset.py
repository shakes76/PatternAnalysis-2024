#!/usr/bin/env python
""" Initialises monai transformations and loads paths to image and label nifti files.

dataset.py loads the files to perform image segmentation. It loads the paths to images and corresponding
labels, but none of them are processed in the file. Moreover, transformation on training set and test set
are defined in the file, but they are to be exported with corresponding dictionary files and used with
monai.data.Dataset and Dataloader.

"""

# ==========================
# Imports
# ==========================
import os
from sklearn.model_selection import train_test_split
from monai.transforms import (LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
                               SpatialCropd, RandFlipd, RandRotated, AsDiscreted,
                               RandGaussianNoised, Compose, CastToTyped, Resized)
import torch

__author__ = "Ryuto Hisamoto"

__license__ = "Apache"
__version__ = "1.0.0"
__maintainer__ = "Ryuto Hisamoto"
__email__ = "s4704935@student.uq.edu.au"
__status__ = "Committed"

# ==========================
# Constants
# ==========================

# IMAGE_FILE_NAME = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs' # on rangpur
# LABEL_FILE_NAME = '/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only' # on rangpur

IMAGE_FILE_NAME = os.path.join(os.getcwd(), 'semantic_MRs_anon') # assuming folders are in the cwd
LABEL_FILE_NAME = os.path.join(os.getcwd(), 'semantic_labels_anon')

rawImageNames = sorted(os.listdir(IMAGE_FILE_NAME))
rawLabelNames = sorted(os.listdir(LABEL_FILE_NAME))

# Split the set into train, validation, and test set (80 : 20 for train:test)
train_images, test_images, train_labels, test_labels = train_test_split(rawImageNames, rawLabelNames, train_size=0.8) # Split the data in training and test set

"""
A transformation is performed in for consistent dimensions in each images and labels, and random augmentation
of files to prevent the model's overfitting to the training set. They are performed in the order of: loading, cropping (to remove extra dimensions),
normalisation of voxel values, random vertical flip (spatial_axis = 2), random rotation (of small degrees), and 
addition of random noise. For labels, an extra step to change encodings is applied.
"""

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialCropd(keys=["image", "label"], roi_slices=[slice(None), slice(None), slice(0, 128)]),  # Crop to depth 128
        NormalizeIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 64)),
        RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        RandRotated(keys=["image", "label"], range_x=0.5, range_y=0.5, range_z=0.5, mode='nearest', prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.5),
        AsDiscreted(keys=["label"], to_onehot=6),
        CastToTyped(keys=["label"], dtype=torch.uint8),
    ]
)

"""
A transformation on test set involved the loading on images and labels, cropping for consistent dimensions,
normalisation of voxel values and encoding of labels.
"""

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialCropd(keys=["image", "label"], roi_slices=[slice(None), slice(None), slice(0, 128)]),
        NormalizeIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 64)),
        AsDiscreted(keys=["label"], to_onehot=6),
        CastToTyped(keys=["label"], dtype=torch.uint8),
    ]
)

# Loads paths to images and labels, but do not process them yet

train_dict = [{"image": os.path.join(IMAGE_FILE_NAME, image), "label": os.path.join(LABEL_FILE_NAME, label)}
               for image, label in zip(train_images, train_labels)]
test_dict = [{"image": os.path.join(IMAGE_FILE_NAME, image), "label": os.path.join(LABEL_FILE_NAME, label)}
              for image, label in zip(test_images, test_labels)]