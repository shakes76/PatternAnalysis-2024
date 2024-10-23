# ==========================
# Imports
# ==========================
import os
from sklearn.model_selection import train_test_split
from monai.transforms import (LoadImaged, EnsureChannelFirstd, RandScaleIntensityd,
                               SpatialCropd, RandFlipd, RandRotated, AsDiscreted,
                               RandGaussianNoised, Compose, CastToTyped)
import torch

# ==========================
# Constants
# ==========================

IMAGE_FILE_NAME = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs'
LABEL_FILE_NAME = '/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only'

# IMAGE_FILE_NAME = os.path.join(os.getcwd(), 'semantic_MRs_anon')
# LABEL_FILE_NAME = os.path.join(os.getcwd(), 'semantic_labels_anon')

rawImageNames = sorted(os.listdir(IMAGE_FILE_NAME))
rawLabelNames = sorted(os.listdir(LABEL_FILE_NAME))

# Split the set into train, validation, and test set (80 : 20 for train:test)
train_images, test_images, train_labels, test_labels = train_test_split(rawImageNames, rawLabelNames, train_size=0.8) # Split the data in training and test set

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialCropd(keys=["image", "label"], roi_slices=[slice(None), slice(None), slice(0, 128)]),  # Crop to depth 128
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandRotated(keys=["image", "label"], range_x=0.5, range_y=0.5, range_z=0.5, mode='nearest', prob=0.5),
        RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0, std=0.5),
        AsDiscreted(keys=["label"], to_onehot=6),
        CastToTyped(keys=["label"], dtype=torch.uint8),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        SpatialCropd(keys=["image", "label"], roi_slices=[slice(None), slice(None), slice(0, 128)]),
        AsDiscreted(keys=["label"], to_onehot=6),
        CastToTyped(keys=["label"], dtype=torch.uint8),
    ]
)

train_dict = [{"image": os.path.join(IMAGE_FILE_NAME, image), "label": os.path.join(LABEL_FILE_NAME, label)} for image, label in zip(train_images, train_labels)]
test_dict = [{"image": os.path.join(IMAGE_FILE_NAME, image), "label": os.path.join(LABEL_FILE_NAME, label)} for image, label in zip(test_images, test_labels)]