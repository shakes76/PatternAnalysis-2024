import numpy as np
import nibabel as nib
import tqdm 

def to_channels(arr: np.ndarray, dtype=np.uint8):
	channels = np.unique(arr)
	res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
	for c in channels:
		c = int(c)
		res[..., c:c+1][arr == c ] = 1
		
	return res

def load_data_3D(image_names, norm_image=False, categorical=False, dtype=np.float32, get_affines=False, orient=False, early_stop=False):
    """
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice 
        thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is 
        masks
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick 
    loading and testing scripts.
    """

    affines = []
    interp = 'linear'
    if dtype == np.uint8:  # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(image_names)
    nifti_image = nib.load(image_names[0])

    first_case = nifti_image.get_fdata(caching='unchanged')

    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0]  # sometimes extra dims, remove

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm.tqdm(image_names)):
        nifti_image = nib.load(inName)

        in_image = nifti_image.get_fdata(caching='unchanged')  # read from disk only
        affine = nifti_image.affine

        if len(in_image.shape) == 4:
            in_image = in_image[:, :, :, 0]  # sometimes extra dims in HipMRI_study data

        in_image = in_image[:, :, :depth]  # clip slices
        in_image = in_image.astype(dtype)

        if norm_image:
            in_image = (in_image - in_image.mean()) / in_image.std()

        if categorical:
            in_image = to_channels(in_image, dtype=dtype)
            images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2], :in_image.shape[3]] = in_image  # with pad
        else:
            images[i, :in_image.shape[0], :in_image.shape[1], :in_image.shape[2]] = in_image  # with pad

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if get_affines:
        return images, affines
    else:
        return images