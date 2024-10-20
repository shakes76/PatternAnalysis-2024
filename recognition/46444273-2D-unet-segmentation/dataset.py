import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pathlib
import tensorflow as tf
from tqdm import tqdm

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res

# load medical image functions
def load_data_2D(imageNames, normImage = False, categorical = False, dtype = np.float32,
    getAffines = False, early_stop = False, image_limit = None):
    '''
    Load medical image data from names , cases list provided into a list for each .

    This function pre - allocates 4 D arrays for conv2d to avoid excessive memory
    usage .

    normImage : bool ( normalise the image 0.0 -1.0)
    early_stop : Stop loading pre - maturely , leaves arrays mostly empty , for quick
    loading and testing scripts .
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # sometimes extra dims , remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, 256, 128, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, 256, 128), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            #inImage = inImage / np . linalg . norm ( inImage )
            #inImage = 255. * inImage / inImage . max ()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            if inImage.shape[1] == 144:
                inImage = inImage[:,:128]
            images[i,:,:] = inImage

        affines.append(affine)
        if i > image_limit and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

# load 2D Nifti data
def get_training_data(image_limit=None):
    early_stop = False

    # training images
    train_data_dir = pathlib.Path('dataset/keras_slices_train').with_suffix('')
    train_image_count = len(list(train_data_dir.glob('*.nii')))
    print(f"test image count: {train_image_count}")
    # training masks
    seg_train_data_dir = pathlib.Path('dataset/keras_slices_seg_train').with_suffix('')
    seg_train_image_count = len(list(seg_train_data_dir.glob('*.nii')))
    print(f"seg test image count: {seg_train_image_count}")
    # testing images
    test_data_dir = pathlib.Path('dataset/keras_slices_test').with_suffix('')
    test_image_count = len(list(test_data_dir.glob('*.nii')))
    print(f"test image count: {test_image_count}")
    # testing masks
    seg_test_data_dir = pathlib.Path('dataset/keras_slices_seg_test').with_suffix('')
    seg_test_image_count = len(list(seg_test_data_dir.glob('*.nii')))
    print(f"seg test image count: {seg_test_image_count}")

    if image_limit is None:
        image_limit = 11460
        early_stop = False
    else:
        early_stop = True
  
    # loading train images
    train_data = load_data_2D(list(train_data_dir.glob('*.nii')), normImage=False,
                              categorical=False, early_stop=early_stop,
                                image_limit=image_limit)[:image_limit,:,:]
    # loading train masks
    seg_train_data = load_data_2D(list(seg_train_data_dir.glob('*.nii')),
                                   normImage=False, categorical=False,
                                   early_stop=early_stop,
                                     image_limit=image_limit).astype(np.uint8)[:image_limit,:,:]
    # loading testing images
    test_data = load_data_2D(list(test_data_dir.glob('*.nii')), normImage=False,
                              categorical=False, early_stop=early_stop,
                                image_limit=image_limit)[:image_limit,:,:]
    # loading testing masks
    seg_test_data = load_data_2D(list(seg_test_data_dir.glob('*.nii')),
                                  normImage=False, categorical=False,
                                    early_stop=early_stop,
                                      image_limit=image_limit).astype(np.uint8)[:image_limit,:,:]

    # expand image data dims
    train_data = np.expand_dims(np.array(train_data), 3)
    test_data = np.expand_dims(np.array(test_data), 3)

    # convert masks to categorical
    n_classes = 6

    from keras.utils import to_categorical
    train_labels = to_categorical(seg_train_data, num_classes=n_classes, dtype=np.uint8)
    train_labels = train_labels.reshape((seg_train_data.shape[0], seg_train_data.shape[1], seg_train_data.shape[2], n_classes))

    test_labels = to_categorical(seg_test_data, num_classes=n_classes, dtype=np.uint8)
    test_labels = test_labels.reshape((seg_test_data.shape[0], seg_test_data.shape[1], seg_test_data.shape[2], n_classes))

    #TODO: Normalize input data

    X_train, X_test, y_train, y_test = train_data, test_data, train_labels, test_labels

    return (X_train, y_train), (X_test, y_test)

def batch_loader(data, batch_size):

    L = len(data[0])

    # keras requires infinite generator
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X_data, y_data = data

            X = X_data[batch_start:limit]
            Y = y_data[batch_start:limit]
            Y = Y.astype(np.float32)

            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size

def data_loader(train_data, test_data, batch_size):
    train_img_datagen = batch_loader(train_data, batch_size)
    test_img_datagen = batch_loader(test_data, batch_size)
    
    return (train_img_datagen, test_img_datagen)
