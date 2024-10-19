import numpy as np
import nibabel as nib
import pathlib
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
    getAffines = False, early_stop = False):
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
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

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
            images[i,:,:] = inImage

        affines.append(affine)
        if i > 100 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

# load 2D Nifti data

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

# loading train images
train_data = load_data_2D(list(train_data_dir.glob('*.nii')), normImage=False, categorical=False, early_stop=True)[:100,:,:]
# loading train masks
seg_train_data = load_data_2D(list(seg_train_data_dir.glob('*.nii')), normImage=False, categorical=False, early_stop=True).astype(np.uint8)[:100,:,:]
# loading testing images
test_data = load_data_2D(list(test_data_dir.glob('*.nii')), normImage=False, categorical=False, early_stop=True)[:100,:,:]
# loading testing masks
seg_test_data = load_data_2D(list(seg_test_data_dir.glob('*.nii')), normImage=False, categorical=False, early_stop=True).astype(np.uint8)[:100,:,:]
