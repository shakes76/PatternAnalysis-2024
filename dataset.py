import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert array to one-hot encoding
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    unique_classes = np.unique(arr)  # Find unique classes
    one_hot = np.zeros(arr.shape + (len(unique_classes),), dtype=dtype)  # Initialize one-hot array
    for c in unique_classes:
        c = int(c)
        one_hot[..., c:c + 1][arr == c] = 1  # Set one-hot encoding

    return one_hot

# Load 2D medical image data
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False):
    '''
    Load medical image data from provided filenames.

    normImage: bool (normalize image to 0.0 - 1.0)
    early_stop: stop loading prematurely for quick tests
    '''

    affines = []  # Store affine transformations

    num = len(imageNames)  # Number of images
    print("Length of Images: ", num)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')  # Load first image
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Take first slice if 3D
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)  # Convert to one-hot if categorical
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)  # Pre-allocate images
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)  # Pre-allocate images

    # Load each image
    for i, inName in enumerate(tqdm(imageNames)):
        try:
            niftiImage = nib.load(inName)  # Load NIfTI image
            inImage = niftiImage.get_fdata(caching='unchanged')  # Get image data
            affine = niftiImage.affine  # Get affine transformation
            if len(inImage.shape) == 3:
                inImage = inImage[:, :, 0]  # Take first slice if 3D
            inImage = inImage.astype(dtype)  # Convert to specified dtype
            if normImage:
                inImage = (inImage - inImage.mean()) / inImage.std()  # Normalize image
            if categorical:
                inImage = to_channels(inImage, dtype=dtype)  # Convert to one-hot if categorical
                images[i, :, :, :] = inImage  # Store in images array
            else:
                images[i, :, :] = inImage  # Store in images array

            affines.append(affine)  # Store affine
            if i > 20 and early_stop:
                break  # Early stop if set
        except:
            print("Error occurred on image: ", i, inName)  # Error handling
            
    if getAffines:
        return images, affines  # Return images and affines if requested
    else:
        return images  # Return only images