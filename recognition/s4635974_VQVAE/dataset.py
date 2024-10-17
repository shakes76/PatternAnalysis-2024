import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.ndimage


# Directories for datasets
train_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
test_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
validate_dir = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'

# Helper function to load NIfTI files
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Convert a 2D array into a one-hot encoded 3D array along a channel axis.
    
    Parameters:
    - arr: 2D numpy array, input image array.
    - dtype: Data type for the resulting array.
    
    Returns:
    - 3D numpy array with channels representing one-hot encoded values.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# Helper function to ensure all images are of the same size
def resize_image(inImage, target_shape):
    """
    Resize a 2D image to the target shape using bilinear interpolation.
    
    Parameters:
    - inImage: Input image to be resized (2D numpy array).
    - target_shape: Tuple (height, width) specifying the target size.
    
    Returns:
    - Resized 2D image.
    """
    # Get the zoom factors for resizing
    zoom_factors = [
        target_shape[i] / inImage.shape[i] for i in range(len(target_shape))
    ]
    # Resize image using the zoom factors
    return scipy.ndimage.zoom(inImage, zoom_factors, order=1)  # Bilinear interpolation

# Load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False, target_shape=(256, 128)):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    Parameters:
    - normImage: bool (normalize the image 0.0-1.0)
    - early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    - target_shape: Target shape to resize the images (rows, cols).
    '''
    affines = []
    
    # Get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Sometimes extra dims, remove
    
    rows, cols = target_shape  # Use the target shape for consistency
    images = np.zeros((num, rows, cols), dtype=dtype)
    
    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # Read from disk only
        affine = niftiImage.affine
        
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        
        # Normalize image if required
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        
        # Resize image to the target shape
        inImage_resized = resize_image(inImage, target_shape)
        
        images[i, :, :] = inImage_resized
        
        affines.append(affine)
        
        if i > 20 and early_stop:
            break
    
    if getAffines:
        return images, affines
    else:
        return images

# Custom Dataset class for loading data into PyTorch
class HipMRIDataset(Dataset):
    """
    Custom Dataset class to load MRI images from a directory for use in PyTorch.

    The images are loaded, optionally transformed, and normalized.
    
    Attributes:
    - image_dir: Directory where MRI images are stored.
    - transform: Optional transform to be applied on each image.
    - normImage: Boolean flag for normalizing images.
    """
    def __init__(self, image_dir, transform=None, normImage=False):
        self.image_dir = image_dir
        self.image_names = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.nii.gz')]
        self.transform = transform
        self.normImage = normImage  # Store the normalization flag

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Load an image, apply transformations, and return it as a tensor.
        
        Parameters:
        - idx: Index of the image to load.

        Returns:
        - Image tensor of shape [C, H, W] where C is the number of channels.
        """
        # Get the image path
        image_path = self.image_names[idx]

        # Use load_data_2D to load the image
        # Since load_data_2D expects a list of image names, wrap the path in a list
        image_data = load_data_2D([image_path], normImage=self.normImage)

        # Extract the 2D array of the first (and only) image
        image = image_data[0]  # load_data_2D returns a list

        # Conditionally add the channel dimension only if no transform is provided
        if self.transform is None:
            image = image[None, :, :]  # Add channel dimension (from [H, W] to [1, H, W])
        
        if self.transform:
            image = self.transform(image)

        return image

# Dataloader class for HipMRI data
class HipMRILoader:
    """
    Class to create DataLoader instances for training, validation, and test datasets.
    
    Attributes:
    - train_loader: DataLoader for the training set.
    - validate_loader: DataLoader for the validation set.
    - test_loader: DataLoader for the test set.
    - train_mean: Mean value of the training data.
    - train_variance: Variance of the training data.
    """
    def __init__(self, train_dir, validate_dir, test_dir, batch_size=16, transform=None, num_workers=1):
        self.batch_size = batch_size
        self.transform = transform
        
        # Create datasets
        self.train_dataset = HipMRIDataset(train_dir, transform=self.transform, normImage=True)
        self.validate_dataset = HipMRIDataset(validate_dir, transform=None, normImage=True)
        self.test_dataset = HipMRIDataset(test_dir, transform=None, normImage=True)  # No transforms for test data
        
        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.validate_loader = DataLoader(self.validate_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # Calculate the variance for data normalization
        self.train_mean, self.train_variance = self.calculate_mean_variance()

    def calculate_mean_variance(self):
        mean = 0.0
        mean_sq = 0.0
        count = 0

        for index, data in enumerate(self.train_loader):
            data = data.float()  # Ensure the data is in the correct type
            mean += data.sum()
            mean_sq += (data ** 2).sum()
            count += np.prod(data.shape)

        total_mean = mean / count
        total_var = (mean_sq / count) - (total_mean ** 2)
        data_variance = float(total_var.item())  # Convert tensor to float
        return total_mean.item(), data_variance

    def get_loaders(self):
        return self.train_loader, self.validate_loader, self.train_variance

    def get_test_loader(self):
        # Create a separate data loader for the test dataset
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        return test_loader
    
    def get_mean(self):
        return self.train_mean