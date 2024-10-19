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


import os

# online uq root
dataroot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/"

testroot = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"

# List all NIfTI files in the folder (ending with .nii.gz)
nifti_files = [os.path.join(testroot, f) for f in os.listdir(testroot) if f.endswith('.nii.gz')]

images = load_data_2D(nifti_files, normImage=True, categorical=False, dtype=np.float32, getAffines=False, early_stop=True)

# After loading, `images` will contain the processed data ready for training your UNet
print(f"Loaded {len(images)} images from {testroot}")



#Test to download an image to ensure nifti files are being read correctly.
import numpy as np
import matplotlib

# Set MPLCONFIGDIR to a writable directory
os.environ['MPLCONFIGDIR'] = '/home/Student/s4838078/matplotlib_cache'

# Verify that MPLCONFIGDIR is set correctly
print(f"MPLCONFIGDIR is set to: {os.environ.get('MPLCONFIGDIR')}")

# Check if the directory exists and create it if needed
if not os.path.exists(os.environ['MPLCONFIGDIR']):
    os.makedirs(os.environ['MPLCONFIGDIR'])

# Import matplotlib after setting MPLCONFIGDIR
import matplotlib.pyplot as plt

print(f"Matplotlib cache directory: {matplotlib.get_cachedir()}")

# Assuming "images" is the array returned by load_data_2D
# Example: images = load_data_2D(nifti_files, normImage=True, categorical=False, dtype=np.float32)

# Select a random index from the images array
random_idx = np.random.randint(0, images.shape[0])

# Get the image at that index (assuming it's a 2D image, or a slice if it's 3D)
random_image = images[random_idx]

# If "random_image" has multiple channels (e.g., for segmentation masks), select the first channel
if len(random_image.shape) > 2:
    random_image = random_image[:, :, 0]  # Modify this depending on the actual image shape

#output_filepath = "/home/Student/s4838078/testimages/random_image_test.png"
# Ensure the directory exists (this will create the directory if it does not exist)
output_dir = "/home/Student/s4838078/testimages"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Full path to the output image
output_filepath = os.path.join(output_dir, "random_image_test.png")

# Save the selected image as a PNG file for download
plt.imshow(random_image, cmap='gray')
plt.axis('off')  # Hide axes for a cleaner image
plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0)
