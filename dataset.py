import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert the array into one-hot encoded channel representation.
    Each unique value in the input becomes its own channel.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    
    return res

def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load 2D medical image data from a list of NIfTI files.
    Args:
        imageNames: List of paths to NIfTI image files.
        normImage: If True, normalize the image.
        categorical: If True, one-hot encode the image based on unique values.
        dtype: Data type for the image arrays.
        getAffines: If True, return affine matrices along with images.
        early_stop: If True, stop after processing 20 images (for testing).
    Returns:
        A numpy array of loaded images and optionally their affines.
    """
    affines = []

    # Get image dimensions from the first image
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # remove extra dimensions if present
    
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # remove extra dimensions
        
        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)
        
        if i > 50 and early_stop:
            break

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

# Example usage
if __name__ == "__main__":
    image_dir = r"C:\Users\舒画\Downloads\HipMRI_study_keras_slices_data\keras_slices_train"
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # Load the images
    images, affines = load_data_2D(image_files, normImage=True, categorical=False, getAffines=True, early_stop=True)

    # Save the first loaded image as an example
    output_path = "output_image.nii.gz"
    save_nifti(images[0], affines[0], output_path)
    print(f"First image saved to {output_path}")
