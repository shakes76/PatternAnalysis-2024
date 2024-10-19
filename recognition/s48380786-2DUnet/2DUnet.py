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
            inImage = utils.to_channels(inImage, dtype=dtype)
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
            inImage = utils.to_channels(inImage, dtype=dtype)
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
# Images are not able to be resized because the error occurs before resizing. load_data_2D expects consistent sizing beforehand
train_images = load_data_2D(train_image_paths, normImage=True, categorical=False)
train_labels = load_data_2D(train_label_paths, normImage=False, categorical=True)
train_images_resized = resize_images_skimage(train_images, target_shape)

val_images = load_data_2D(val_image_paths, normImage=True, categorical=False)
val_labels = load_data_2D(val_label_paths, normImage=False, categorical=True)
val_images_resized = resize_images_skimage(val_images, target_shape)

test_images = load_data_2D(test_image_paths, normImage=True, categorical=False)
test_labels = load_data_2D(test_label_paths, normImage=False, categorical=True)
test_images_resized = resize_images_skimage(test_images, target_shape)

