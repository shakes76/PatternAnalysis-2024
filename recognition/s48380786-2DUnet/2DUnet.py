# Reference Link:
# https://www.kaggle.com/code/mrmohammadi/2d-unet-pytorch


import numpy as np
import nibabel as nib
from tqdm import tqdm

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            #inImage = utils.to_channels(inImage, dtype=dtype)
            inImage = to_channels(inImage, dtype=dtype)
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


def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage : bool (normalize the image 0.0-1.0)
    orient : Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    dtype : Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop : Stop loading prematurely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        # testResultName = "oriented.nii.gz"
        # niftiImage.to_filename(testResultName)
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        if orient:
            niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth]  # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            # inImage = inImage / np.linalg.norm(inImage)
            # inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            #inImage = utils.to_channels(inImage, dtype=dtype)
            inImage = to_channels(inImage, dtype=dtype)
            # images[i, :, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage  # with pad
        else:
            # images[i, :, :, :] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage  # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

# Staff linked this page regarding resizing images
# https://stackoverflow.com/questions/64674612/how-to-resize-a-nifti-nii-gz-medical-image-file
import skimage.transform as skTrans

# Define the target shape
target_shape = (256, 128)

"""
# Function to resize images using Scikit-Image
def resize_images_skimage(images, target_shape):
    # Create an empty array to store the resized images
    #                         ((num images    ,      256       ,        128     )
    resized_images = np.zeros((images.shape[0], target_shape[0], target_shape[1]), dtype=images.dtype)
    
    # Loop through each image in the dataset
    for i, image in enumerate(images):
        # Resize the image to the target shape using Scikit-Image's resize function
        resized_image = skTrans.resize(image, target_shape, order=1, preserve_range=True, anti_aliasing=True)

        # Store the resized image in the resized_images array
        resized_images[i, :, :] = resized_image

    return resized_images
"""

# Function to resize a single image
def resize_image(image, target_shape):
    return skTrans.resize(image, target_shape, order=1, preserve_range=True, anti_aliasing=True)

def pad_channels(image, target_channels):
    """
    Pad the channels of an image to the target number of channels.
    """
    current_channels = image.shape[-1]
    
    # If the current number of channels is less than the target, pad with zeros
    if current_channels < target_channels:
        padding_shape = list(image.shape)
        padding_shape[-1] = target_channels - current_channels
        padding = np.zeros(padding_shape, dtype=image.dtype)
        image = np.concatenate((image, padding), axis=-1)

    return image

# Function to load and resize images one by one using load_data_2D
def load_and_resize_images(image_paths, target_shape, normImage=False, categorical=False, target_channels=6):
    resized_images = []  # To store resized images

    for image_path in image_paths:
        # Load image one at a time using load_data_2D
        image = load_data_2D([image_path], normImage=normImage, categorical=categorical)  # Loading one image at a time
        resized_image = resize_image(image[0], target_shape)  # Resize the single image

        # If categorical, pad the channels to the target number of channels
        if categorical:
            resized_image = pad_channels(resized_image, target_channels)

        resized_images.append(resized_image)  # Append the resized image to the list

    # Stack all resized images into a NumPy array
    return np.stack(resized_images)

import os
# Define the root directory
dataroot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"


# Create paths for images and segmentation labels
train_image_dir = os.path.join(dataroot, "keras_slices_train")
train_label_dir = os.path.join(dataroot, "keras_slices_seg_train")

val_image_dir = os.path.join(dataroot, "keras_slices_validate")
val_label_dir = os.path.join(dataroot, "keras_slices_seg_validate")

test_image_dir = os.path.join(dataroot, "keras_slices_test")
test_label_dir = os.path.join(dataroot, "keras_slices_seg_test")

# Get all image and label file paths
train_image_paths = sorted([os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith('.nii.gz')])
train_label_paths = sorted([os.path.join(train_label_dir, f) for f in os.listdir(train_label_dir) if f.endswith('.nii.gz')])

val_image_paths = sorted([os.path.join(val_image_dir, f) for f in os.listdir(val_image_dir) if f.endswith('.nii.gz')])
val_label_paths = sorted([os.path.join(val_label_dir, f) for f in os.listdir(val_label_dir) if f.endswith('.nii.gz')])

test_image_paths = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith('.nii.gz')])
test_label_paths = sorted([os.path.join(test_label_dir, f) for f in os.listdir(test_label_dir) if f.endswith('.nii.gz')])

# Load data
train_images_resized = load_and_resize_images(train_image_paths, target_shape, normImage=True, categorical=False)
val_images_resized = load_and_resize_images(val_image_paths, target_shape, normImage=True, categorical=False)
test_images_resized = load_and_resize_images(test_image_paths, target_shape, normImage=True, categorical=False)

# Segmentation Masks
train_labels_resized = load_and_resize_images(train_label_paths, target_shape, normImage=False, categorical=True)
val_labels_resized = load_and_resize_images(val_label_paths, target_shape, normImage=False, categorical=True)
test_labels_resized = load_and_resize_images(test_label_paths, target_shape, normImage=False, categorical=True)


#debug
print(f"Resized shape of train_images: {train_images_resized.shape}")
print(f"Resized shape of train_labels: {train_labels_resized.shape}")
print(f"Resized shape of val_images: {val_images_resized.shape}")
print(f"Resized shape of val_labels: {val_labels_resized.shape}")
print(f"Resized shape of test_images: {test_images_resized.shape}")
print(f"Resized shape of test_labels: {test_labels_resized.shape}")


"""
def find_max_channels(image_paths):
    max_channels = 0
    for image_path in image_paths:
        # Load the image (segmentation mask)
        image = load_data_2D([image_path], normImage=False, categorical=False)  # Load without one-hot encoding
        unique_labels = np.unique(image[0])  # Find unique labels in the mask
        max_channels = max(max_channels, len(unique_labels))  # Track the maximum number of unique labels
    return max_channels

# Check maximum number of channels in the training labels
max_train_channels = find_max_channels(train_label_paths)
print(f"Maximum number of channels in training labels: {max_train_channels}")

# Check maximum number of channels in the validation labels
max_val_channels = find_max_channels(val_label_paths)
print(f"Maximum number of channels in validation labels: {max_val_channels}")

# Check maximum number of channels in the test labels
max_test_channels = find_max_channels(test_label_paths)
print(f"Maximum number of channels in test labels: {max_test_channels}")
"""