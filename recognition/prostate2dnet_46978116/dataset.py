import numpy as np
import nibabel as nib
from tqdm import tqdm

def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Load medical image functions
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False
):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    Parameters:
    - imageNames: List of image file paths.
    - normImage: bool (normalize the image to 0.0 - 1.0).
    - categorical: bool (convert to categorical channels).
    - dtype: data type for the images.
    - getAffines: bool (return affine transformations).
    - early_stop: bool (stop loading prematurely for quick loading and testing).

    Returns:
    - images: Loaded image data as a NumPy array.
    - affines (optional): List of affine transformations if getAffines is True.
    '''
    affines = []

    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            # Normalize the image
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


def load_single_data_2D(
    imagePath,
    maskPath,
    normImage=False,
    categorical=False,
    dtype=np.float32
):
    '''
    Load a single pair of image and mask.

    Parameters:
    - imagePath: Path to the image file.
    - maskPath: Path to the mask file.
    - normImage: bool (normalize the image to 0.0- 1.0).
    - categorical: bool (convert to categorical channels).
    - dtype: data type for the images.

    Returns:
    - image: Loaded and processed image as a NumPy array.
    - mask: Loaded and processed mask as a NumPy array.
    '''
    # Load image
    niftiImage = nib.load(imagePath)
    image = niftiImage.get_fdata(caching='unchanged')
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Remove extra dimensions
    image = image.astype(dtype)
    if normImage:
        # Normalize the image to have zero mean and unit variance
        image = (image - image.mean()) / image.std()

    # Load mask
    niftiMask = nib.load(maskPath)
    mask = niftiMask.get_fdata(caching='unchanged')
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Remove extra dimensions
    mask = mask.astype(dtype)

    if categorical:
        mask = to_channels(mask, dtype=np.uint8)
    else:
        mask = (mask > 0).astype(np.float32)  # Binary mask

    return image, mask

class ProstateMRIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, norm=True, categorical=False, early_stop=False):
        """
        Initializes the dataset by loading all images and masks into memory.

        Parameters:
        - image_dir (str): Directory containing image Nifti files.
        - mask_dir (str): Directory containing mask Nifti files.
        - transform (callable, optional): Optional transform to be applied on a sample.
        - norm (bool): Whether to normalize the images.
        - categorical (bool): Whether masks are categorical (multi-class).
        - early_stop (bool): If True, stops loading after 20 samples for quick testing.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.norm = norm
        self.categorical = categorical
        self.early_stop = early_stop

        # Retrieve sorted lists of image and mask filenames
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be equal."

        # Full paths
        self.image_paths = [os.path.join(image_dir, f) for f in self.image_files]
        self.mask_paths = [os.path.join(mask_dir, f) for f in self.mask_files]

        # Early stop handling
        if self.early_stop:
            self.image_paths = self.image_paths[:20]
            self.mask_paths = self.mask_paths[:20]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the specified index.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (torch.Tensor): Image tensor.
        - mask (torch.Tensor): Corresponding mask tensor.
        """
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image, mask = load_single_data_2D(
            imagePath=image_path,
            maskPath=mask_path,
            normImage=self.norm,
            categorical=self.categorical,
            dtype=np.float32
        )

        # Convert image to tensor
        image = torch.from_numpy(image).unsqueeze(0).float() 
        if self.categorical:
 
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()  
    
            mask = torch.from_numpy(mask).unsqueeze(0).float() 

        if self.transform:
            image = self.transform(image)

        return image, mask