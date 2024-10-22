import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import zoom
import torchio as tio
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Function to load 3D medical image data from a list of file paths
def load_data_3D(image_names, normalize=False, dtype=np.float32, target_shape=(96, 96, 96), augment=False):
    '''
    Load 3D medical image data from file paths into an array.

    Parameters:
    - image_names: List of file paths to Nifti files.
    - normalize: bool (default=False). Normalize the image to [0, 1] or mean 0, std 1.
    - dtype: Data type for the loaded images (default=np.float32). If dtype=np.uint8, data is assumed to be labels.
    - target_shape: tuple (default=(96, 96, 96)). The target shape to ensure all images have the same dimensions.
    - augment: bool (default=False). Perform data augmentation if True.

    Returns:
    - 3D images in NumPy array format.
    '''
    num_images = len(image_names)

    # Pre-allocate memory for the images with the specified target shape
    rows, cols, depth = target_shape
    images = np.zeros((num_images, rows, cols, depth), dtype=dtype)

    # Initialize augmenter if augmentation is requested
    augmenter = Augment(target_shape=target_shape) if augment else None

    # Iterate over each image file and load data
    for i, image_path in enumerate(tqdm(image_names)):
        nifti_image = nib.load(image_path)

        # Extract image data from Nifti object
        image_data = nifti_image.get_fdata()

        # Resize the image to the target shape
        image_data = resize_image(image_data, target_shape)

        # Apply data augmentation if requested
        if augmenter:
            image_data = augmenter.apply_augmentation(image_data)

        # Convert the image data to the desired type
        image_data = image_data.astype(dtype)

        # Normalize the image data if necessary (only for non-label data)
        if normalize and dtype != np.uint8:
            image_data = (image_data - image_data.mean()) / (image_data.std() + 1e-8)

        # Store the processed image in the pre-allocated array
        images[i, :, :, :] = image_data

    return images


# Function to resize a 3D image to the target shape using interpolation
def resize_image(image, target_shape):
    '''
    Resize a 3D image to the target shape using zoom interpolation.
    '''
    current_shape = image.shape
    scale_factors = [target / current for target, current in zip(target_shape, current_shape)]
    resized_image = zoom(image, scale_factors, order=1)  # Use linear interpolation for resizing
    return resized_image


# Class for basic data augmentation, such as random flipping
class Augment:
    def __init__(self, target_shape=(64, 64, 64)):
        # Define the augmentation methods and their probabilities
        flip = tio.transforms.RandomFlip()  # No flip
        bias_field = tio.transforms.RandomBiasField()  # Bias field distortion
        blur = tio.transforms.RandomBlur()  # Random blur
        spike = tio.transforms.RandomSpike()  # MRI artifact spikes
        gamma = tio.transforms.RandomGamma(log_gamma=(-0.3, 0.3))  # Random gamma adjustment
        elastic = tio.transforms.RandomElasticDeformation(num_control_points=7, max_displacement=7.5)  # Elastic deformation

        # Define probabilities for each augmentation method
        prob = {flip: 0.5, bias_field: 0.1, blur: 0.1, spike: 0.1, gamma: 0.1, elastic: 0.1}

        # Apply a random augmentation based on the defined probabilities
        self.oneof = tio.transforms.OneOf(prob)

        # Use multiple augmentations sequentially for weak classes
        self.heavy_augment = tio.Compose([
            tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),  # Random flipping along all axes
            tio.RandomAffine(scales=(0.8, 1.2), degrees=(-15, 15), isotropic=True),  # Random affine transform
            tio.RandomGamma(log_gamma=(-0.3, 0.3)),  # Gamma adjustment
            tio.RandomElasticDeformation(num_control_points=7, max_displacement=8.0),  # Elastic deformation
            tio.RandomNoise(mean=0, std=0.05),  # Add random noise
            tio.CropOrPad(target_shape),  # Crop or pad to the target shape
        ])

        # Crop or pad the image to the target shape
        self.shrink = tio.CropOrPad(target_shape)

    def apply_augmentation(self, image, label):
        """
        Apply augmentation only if the label contains class 4 or 5 (weaker classes).

        Parameters:
        - image: 3D medical image (NumPy array).
        - label: 3D label image (NumPy array).

        Returns:
        - Augmented image if label contains class 4 or 5, otherwise returns the original image.
        """
        # Check if label contains class 4 or 5 (weak classes)
        if np.isin([4, 5], label).any():
            # Convert the image to a TorchIO ScalarImage for augmentation
            image = tio.ScalarImage(tensor=torch.tensor(image).unsqueeze(0))  # Add a channel dimension

            # First, crop or pad the image to the target shape
            image = self.shrink(image)

            # Apply the heavier augmentation sequence for weak classes
            image = self.heavy_augment(image)

            # Convert the augmented image back to a NumPy array
            image = image.data.numpy()

            # Remove the channel dimension for compatibility with the rest of the pipeline
            image = np.squeeze(image)
        else:
            # If not a weak class, apply the standard augmentation
            image = tio.ScalarImage(tensor=torch.tensor(image).unsqueeze(0))
            image = self.shrink(image)
            image = self.oneof(image)
            image = image.data.numpy()
            image = np.squeeze(image)

        return image


# Custom Dataset class for handling MRI and label data
class ProstateMRI3DDataset(Dataset):
    def __init__(self, mri_dir, labels_dir, target_shape=(96, 96, 96), augment=True):
        '''
        Initialize the dataset with directories containing MRI and label data.
        '''
        self.mri_files = sorted(os.listdir(mri_dir))
        self.label_files = sorted(os.listdir(labels_dir))
        self.mri_dir = mri_dir
        self.labels_dir = labels_dir
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.mri_files)

    def __getitem__(self, idx):
        '''
        Load the MRI and label data for a specific index.
        '''
        mri_path = os.path.join(self.mri_dir, self.mri_files[idx])
        label_path = os.path.join(self.labels_dir, self.label_files[idx])

        mri_data = load_data_3D([mri_path], normalize=True, target_shape=self.target_shape, augment=self.augment)
        label_data = load_data_3D([label_path], target_shape=self.target_shape, dtype=np.uint8)

        mri_data = mri_data[0]
        label_data = label_data[0]

        return torch.tensor(mri_data.copy(), dtype=torch.float32), torch.tensor(label_data.copy(), dtype=torch.long)


# Initialize directories and parameters
MRI_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
LABEL_DIR = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
TARGET_SHAPE = (96, 96, 96)
BATCH_SIZE = 4

# Create dataset object
dataset = ProstateMRI3DDataset(MRI_DIR, LABEL_DIR, target_shape=TARGET_SHAPE, augment=True)

# Split dataset into training, validation, and testing sets (90%, 5%, 5%)
train_size = int(0.9 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

