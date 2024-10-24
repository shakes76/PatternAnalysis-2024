import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize
import glob
import tensorflow as tf


def to_channels(arr: np.ndarray , dtype = np.uint8) -> np.ndarray :
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (6,), dtype = dtype)
    for c in channels:
        c1 = int(c)
        res[..., c1:c1+1][arr == c] = 1
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False, target_shape=(256, 128)):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage : bool (normalise the image 0.0-1.0)
    early_stop : Stop loading prematurely, leaves arrays mostly empty, for quick loading and testing scripts.
    target_shape : tuple (height, width) to resize images to a consistent size
    """
    affines = []

    # Get fixed size 
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


    # Load each image
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Remove extra dims
        inImage = inImage.astype(dtype)

        # Normalize the image if required
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        # Resize image to target shape
        inImage = resize(inImage, target_shape, anti_aliasing=True)



        if categorical:
            #print("Unique classes in segmentation mask:", np.unique(inImage), inImage.shape)
            inImage = to_channels(inImage, dtype=dtype)
            images [i ,: ,: ,:] = inImage
        else:
            images [i ,: ,:] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images


def load_data():
    # Get all file paths for train, test, and validate sets
    #train_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_train/*.nii.gz')
    #test_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_test/*.nii.gz')
    #validate_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_validate/*.nii.gz')

    #seg_train_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_train/*.nii.gz')
    #seg_test_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_test/*.nii.gz')
    #seg_validate_files = glob.glob(r'C:\Users\jackr\Desktop\HipMRI_study_keras_slices_data\keras_slices_seg_validate/*.nii.gz')

    train_files = sorted(glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train/*.nii.gz'))
    test_files = sorted(glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test/*.nii.gz'))
    validate_files = sorted(glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate/*.nii.gz'))

    seg_train_files = sorted(glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train/*.nii.gz'))
    seg_test_files = sorted(glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test/*.nii.gz'))
    seg_validate_files = sorted(glob.glob(r'/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate/*.nii.gz'))

    early = False

    # Load the images using the load_data_2D function
    images_train = load_data_2D(train_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128), early_stop=early)
    images_test = load_data_2D(test_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128), early_stop=early)
    images_validate = load_data_2D(validate_files, normImage=True, categorical=False, dtype=np.float32, target_shape=(256, 128), early_stop=early)

    images_seg_train = load_data_2D(seg_train_files, normImage=True, categorical=True, dtype=np.float32, target_shape=(256, 128), early_stop=early)
    images_seg_test = load_data_2D(seg_test_files, normImage=True, categorical=True, dtype=np.float32, target_shape=(256, 128), early_stop=early)
    images_seg_validate = load_data_2D(seg_validate_files, normImage=True, categorical=True, dtype=np.float32, target_shape=(256, 128), early_stop=early)

    # print the shapes of the loaded datasets
    print(f"Training data shape: {images_train.shape}")
    print(f"Test data shape: {images_test.shape}")
    print(f"Validation data shape: {images_validate.shape}")
    print(f"Segement Training data shape: {images_seg_train.shape}")
    print(f"Segement Test data shape: {images_seg_test.shape}")
    print(f"Segement Validation data shape: {images_seg_validate.shape}")


    return images_train, images_test, images_validate, images_seg_test, images_seg_train, images_seg_validate



def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.
    
    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.
    
    Parameters:
    - normImage: bool (normalize the image 0.0-1.0)
    - orient: Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    - dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    - early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    
    affines = []
    #~ interp = ' continuous '
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # Get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    
    if orient:
        niftiImage = im.applyOrientation(niftiImage, interpolation=interp, scale=1)
        #~ testResultName = " oriented . nii .gz"
        #~ niftiImage . to_filename ( testResultName )
    
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
            #~ inImage = inImage / np. linalg . norm ( inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = utils.to_channels(inImage, dtype=dtype)
            #~ images [i ,: ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2], :inImage.shape[3]] = inImage  # with pad
        else:
            #~ images [i ,: ,: ,:] = inImage
            images[i, :inImage.shape[0], :inImage.shape[1], :inImage.shape[2]] = inImage  # with pad

        affines.append(affine)
        
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images