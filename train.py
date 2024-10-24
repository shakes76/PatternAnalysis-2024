import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom  # Used to scale the image


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert the array into one-hot encoded channel representation.
    Each unique value in the input becomes its own channel.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)

    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1

    return res


def resize_image(image, target_shape=(256, 256)):
    """
    Resize a 2D image to the target shape using zoom interpolation.
    Args:
        image: Input 2D image.
        target_shape: Target shape to resize the image.
    Returns:
        Resized image.
    """
    zoom_factors = [t / i for t, i in zip(target_shape, image.shape)]
    return zoom(image, zoom_factors, order=1)  # Bilinear interpolation


def load_data_2D(imageNames, normImage=False, categorical=True, dtype=np.float32, getAffines=False, early_stop=False,
                 target_shape=(256, 256)):
    """
    Load 2D medical image data from a list of NIfTI files.
    Args:
        imageNames: List of paths to NIfTI image files.
        normImage: If True, normalize the image.
        categorical: If True, one-hot encode the image based on unique values.
        dtype: Data type for the image arrays.
        getAffines: If True, return affine matrices along with images.
        early_stop: If True, stop after processing 20 images (for testing).
        target_shape: The target shape to resize all images to.
    Returns:
        A numpy array of loaded images and optionally their affines.
    """
    affines = []
    num = len(imageNames)

    # Initialize and store the adjusted image array
    if categorical:
        channels = len(np.unique(nib.load(imageNames[0]).get_fdata(caching='unchanged')))
        images = np.zeros((num, target_shape[0], target_shape[1], channels), dtype=dtype)
    else:
        images = np.zeros((num, target_shape[0], target_shape[1]), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Take the first layer as the 2D image

        inImage = inImage.astype(dtype)

        # Resize the image
        inImage = resize_image(inImage, target_shape)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

    if getAffines:
        return images, affines
    else:
        return images


def save_nifti(image, affine, out_path):
    """
    Save a 2D or 3D image as a NIfTI file.
    Args:
        image: numpy array containing image data.
        affine: affine matrix for the NIfTI image.
        out_path: Output file path.
    """
    nifti_img = nib.Nifti1Image(image, affine)
    nib.save(nifti_img, out_path)


def load_all_data(image_dir, normImage=False, categorical=False, dtype=np.float32, target_shape=(256, 256)):
    """
    Load all 2D images from a directory containing NIfTI files.
    Args:
        image_dir: Directory containing NIfTI files.
        normImage: If True, normalize the image.
        categorical: If True, one-hot encode the image based on unique values.
        dtype: Data type for the image arrays.
        target_shape: The target shape to resize all images to.
    Returns:
        A numpy array of all loaded images.
    """
    # Get the paths of all NIfTI files in the directory
    imageNames = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                  f.endswith('.nii') or f.endswith('.nii.gz')]
    return load_data_2D(imageNames, normImage=normImage, categorical=categorical, dtype=dtype,
                        target_shape=target_shape)


    # Example usage
if __name__ == "__main__":
    image_dir = r"C:\Users\舒画\Downloads\HipMRI_study_keras_slices_data\keras_slices_train"
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                   f.endswith('.nii') or f.endswith('.nii.gz')]

    # Load the images
    images, affines = load_data_2D(image_files, normImage=True, categorical=False, getAffines=True, early_stop=True,
                                   target_shape=(256, 256))
