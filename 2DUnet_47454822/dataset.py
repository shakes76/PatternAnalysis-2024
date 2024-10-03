# The dataloader
# Loads and preprocesses the data
import os

import numpy as np
import nibabel as nib
from keras.src.utils import to_categorical, load_img, img_to_array
from tqdm import tqdm
def to_channels ( arr : np . ndarray , dtype = np . uint8 ) -> np . ndarray :
    channels = np . unique ( arr )
    res = np . zeros ( arr . shape + ( len ( channels ) ,) , dtype = dtype )
    for c in channels :
        c = int ( c )
        res [... , c : c +1][ arr == c ] = 1
    return res


# load medical image functions
def load_data_2D ( imageNames , normImage = False , categorical = False , dtype = np . float32 , getAffines = False , early_stop = False ) :
    '''
    Load medical image data from names , cases list provided into a list for each .
    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory usage .
    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick loading and testing scripts .
    '''
    affines = []
    # get fixed size
    num = len ( imageNames )
    first_case = nib.load( imageNames [0]).get_fdata (caching = 'unchanged')
    if len ( first_case . shape ) == 3:
        first_case = first_case[:, :, 0]  # sometimes extra dims , remove

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
            # ~ inImage = inImage / np . linalg . norm ( inImage )
            # ~ inImage = 255. * inImage / inImage . max ()

            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
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

def load_dir(image_folder, normImage = False , categorical = False , dtype = np . float32 , getAffines = False , early_stop = False ):
    image_filenames = sorted([image_folder+f for f in os.listdir(image_folder)])
    return load_data_2D(image_filenames, normImage, categorical, dtype, getAffines, early_stop)



#
#
# # ============== Vars ==============
#
# FULL_SIZE_IMG = 1  # set to 2 to use full size image
# INPUT_SHAPE = (32, 32, 3)
# num_classes = 4  # numb of classes in segmentation
#
# # ============== Get Data ==============
#
# def load_and_preprocess_image(image_path, target_size):
#     """Load an image, resize it, and normalize it."""
#     image = load_img(image_path, target_size=target_size, color_mode='grayscale')
#     image = img_to_array(image) / 255.0  # Normalize to [0, 1]
#     return image
#
#
# def load_data(image_folder, mask_folder, target_size):
#     """Load and preprocess images and masks."""
#     # sorted takes in an array.
#     # the for loop creates an array with all the png names in a folder.
#     image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith('.png')])
#     mask_filenames = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])
#
#     images = []
#     masks = []
#
#     for img_name, mask_name in zip(image_filenames, mask_filenames):
#         img_path = os.path.join(image_folder, img_name)
#         mask_path = os.path.join(mask_folder, mask_name)
#
#         image = load_and_preprocess_image(img_path, target_size)
#         mask = load_and_preprocess_image(mask_path, target_size)
#
#         images.append(image)
#         masks.append(mask)
#
#     return np.array(images), np.array(masks)
#
# # Set the target size of the images
#
# target_size = (128*FULL_SIZE_IMG, 128*FULL_SIZE_IMG)
#
# # training data
# train_images, train_masks = load_data('keras_png_slices_data/train/keras_png_slices_train', 'keras_png_slices_data/train/keras_png_slices_seg_train', target_size)
#
# # testing data
# test_images, test_masks = load_data('keras_png_slices_data/test/keras_png_slices_test', 'keras_png_slices_data/test/keras_png_slices_seg_test', target_size)
#
# # For binary masks
# train_masks = train_masks.astype(np.float32)
# test_masks = test_masks.astype(np.float32)
#
# # Convert masks to categorical one-hot encodings
# train_masks = to_categorical(train_masks, num_classes=num_classes)
# test_masks = to_categorical(test_masks, num_classes=num_classes)
#
