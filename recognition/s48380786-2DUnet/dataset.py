import numpy as np
import nibabel as nib
from tqdm import tqdm
import utils
import skimage.transform as skTrans
import torch
from torch.utils.data import DataLoader, TensorDataset
import os


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
        first_case = utils.to_channels(first_case, dtype=dtype)
        #first_case = to_channels(first_case, dtype=dtype)
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
            inImage = utils.to_channels(inImage, dtype=dtype)
            #inImage = to_channels(inImage, dtype=dtype)
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


# Staff linked this page regarding resizing images
# https://stackoverflow.com/questions/64674612/how-to-resize-a-nifti-nii-gz-medical-image-file

# Define the root directory
dataroot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"
# Define the target shape
target_shape = (256, 128)

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
        image = load_data_2D([image_path], normImage=normImage, categorical=categorical, early_stop=False)  # Loading one image at a time
        resized_image = resize_image(image[0], target_shape)  # Resize the single image

        # If categorical, pad the channels to the target number of channels
        if categorical:
            resized_image = pad_channels(resized_image, target_channels)

        resized_images.append(resized_image)  # Append the resized image to the list

    # Stack all resized images into a NumPy array
    return np.stack(resized_images)


def load_and_preprocess_data():
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



    # Convert ===Resized=== NumPy arrays to PyTorch tensors
    train_images_tensor = torch.Tensor(train_images_resized).unsqueeze(1)  # Add channel dimension
    train_labels_tensor = torch.Tensor(train_labels_resized).unsqueeze(1)  # Add channel dimension

    val_images_tensor = torch.Tensor(val_images_resized).unsqueeze(1)   # Add channel dimension
    val_labels_tensor = torch.Tensor(val_labels_resized).unsqueeze(1)   # Add channel dimension

    test_images_tensor = torch.Tensor(test_images_resized).unsqueeze(1)  # Add channel dimension
    test_labels_tensor = torch.Tensor(test_labels_resized).unsqueeze(1)  # Add channel dimension 


    # Create DataLoaders for batching
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, val_loader, test_loader