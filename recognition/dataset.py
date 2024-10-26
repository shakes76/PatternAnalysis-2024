import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib as plt

# Convert an array into categorical channels
def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """
    Converts a 2D array with class labels into a one-hot encoded 3D array of channels.
    Parameters:
        arr (np.ndarray): The input array with class labels.
        dtype (numpy data type): Desired data type of the output array.
    Returns:
        np.ndarray: A 3D array where each channel corresponds to a one-hot encoding of a class.
    """
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c + 1][arr == c] = 1
    return res

# Load medical images with optional categorical and affine handling
def load_data_2D(imageNames, target_shape=(256, 128), normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):
    """
    Load and preprocess 2D medical images, with options for normalization and categorical encoding.
    Parameters:
        imageNames (list): List of file paths to the NIfTI images.
        target_shape (tuple): Desired (height, width) of the images.
        normImage (bool): If True, normalize images to mean=0 and std=1.
        categorical (bool): If True, convert labels to one-hot categorical format.
        dtype (numpy data type): Desired data type for images.
        getAffines (bool): If True, return image affine matrices.
        early_stop (bool): If True, load only a subset of images for quick testing.
    Returns:
        tuple: Array of preprocessed images and (optionally) a list of affine matrices.
    """
    affines = []
    # Determine the number of images and pre-allocate arrays
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Drop extra dimension if present
    # Handle categorical conversion for first image
    rows, cols = target_shape
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        fc_rows, fc_cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        fc_rows, fc_cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    # Load and process each image
    for i, inName in enumerate(tqdm(imageNames, desc="Loading images")):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Ensure 2D by removing extra dimension if needed
        inImage = inImage.astype(dtype)
        # Normalize image if specified
        if normImage:
            inImage = (inImage - inImage.mean()) / inImage.std()
        # Resize if needed
        if inImage.shape != target_shape:
            print("IMAGE RESIZED FROM ORIGINAL SHAPE OF", inImage.shape)
            inImage = resize(inImage, target_shape, mode='constant', preserve_range=True)
        # Convert to categorical format if specified
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage
        # Collect the affine matrix
        affines.append(affine)
        # Stop early if specified
        if i > 20 and early_stop:
            break
    # Return images and affines
    return (images, affines)

# Dataset Class
class SegmentationData(Dataset):
    # Override existing dataset class init function
    def __init__(self, image_dir, label_dir, norm_image=False, categorical=False, dtype=np.float32):
        self.image_filenames = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.label_filenames = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        print(f"Image Dir {image_dir}: {len(self.image_filenames)}")
        print(f"Label Dir {label_dir}: {len(self.label_filenames)}")
        self.images, _ = load_data_2D(self.image_filenames, normImage=norm_image, categorical=False, dtype=dtype)
        self.labels, _ = load_data_2D(self.label_filenames, normImage=False, categorical=categorical, dtype=dtype)
        self.labels = (self.labels > 0).astype(np.float32)

    # Override existing dataset function len
    def __len__(self):
        return len(self.image_filenames)

    # Override existing dataset function getitem
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image[np.newaxis, :], label[np.newaxis, :]
    
    # Plot the image
    def plot_img(self, idx):
        image, label = self[idx]  # Get the image and label using __getitem__
        # Remove extra dimensions for plotting
        image = image[0]  
        label = label[0] 
        height, width = image.shape
        print(f"Image Resolution: {width}x{height}")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Image")
        ax[0].axis('off')
        ax[1].imshow(label, cmap='gray')
        ax[1].set_title("Label")
        ax[1].axis('off')
        plt.show()
    
    # Plot all of the different labels
    def plot_labels(self, idx):
        image, _ = self[idx]  # Get the image using __getitem__
        image = image[0]  # Remove the extra dimension for plotting
        # Set up plotting grid based on the number of labels
        num_labels = len(self.label_filenames)
        fig, axes = plt.subplots(1, num_labels + 1, figsize=(5 * (num_labels + 1), 5))
        # Plot the original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        print(f"Image resolution: {image.shape[:2]}")
        # Plot each label with its resolution and name
        for i, label_name in enumerate(self.label_filenames):
            label = self.labels[idx][i] if len(self.labels.shape) > 3 else self.labels[idx]
            label_resolution = label.shape[:2]
            # Plot the label
            axes[i + 1].imshow(label, cmap='gray')
            axes[i + 1].set_title(f"Label: {os.path.basename(label_name)}")
            axes[i + 1].axis('off')
            # Print label resolution and name
            print(f"Label {i + 1} ({os.path.basename(label_name)}) resolution: {label_resolution}")
        plt.tight_layout()
        plt.show()

    # Plot everything
    def plot_image_and_labels(self, idx):
        self.plot_img(idx)
        self.plot_img_labels(idx)
