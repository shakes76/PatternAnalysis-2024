# DATASET.PY

# IMPORTS
import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Convert an array into categorical channels
def to_channels(arr: np.ndarray, num_classes: int, dtype=np.uint8) -> np.ndarray:
    """
    Converts a 2D array with class labels into a one-hot encoded 3D array of channels.

    Parameters:
        arr (np.ndarray): The input array with class labels.
        num_classes (int): Total number of classes in the dataset.
        dtype (numpy data type): Desired data type of the output array.

    Returns:
        np.ndarray: A 3D array where each channel corresponds to a one-hot encoding of a class.
    """
    res = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    for c in range(num_classes):
        res[..., c][arr == c] = 1
    return res

# Load medical images with optional categorical and affine handling
def load_data_2D(imageNames, target_shape=(256, 128), normImage=False, categorical=False, dtype=np.float32,
                 getAffines=False, early_stop=False, num_classes=None):
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
        num_classes (int): Total number of classes in the dataset (required if categorical=True).

    Returns:
        tuple: Array of preprocessed images and (optionally) a list of affine matrices.
    """
    affines = []
    # Determine the number of images and pre-allocate arrays
    num = len(imageNames)
    rows, cols = target_shape

    # If categorical, num_classes must be specified
    if categorical and num_classes is None:
        raise ValueError("num_classes must be specified when categorical=True")

    # Initialize the images array
    if categorical:
        images = np.zeros((num, rows, cols, num_classes), dtype=dtype)
    else:
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
            print(f"IMAGE RESIZED FROM ORIGINAL SHAPE OF {inImage.shape} to {target_shape}")
            inImage = resize(inImage, target_shape, mode='constant', preserve_range=True, anti_aliasing=True)
        # Convert to categorical format if specified
        if categorical:
            inImage = to_channels(inImage, num_classes=num_classes, dtype=dtype)
            images[i] = inImage
        else:
            images[i] = inImage
        # Collect the affine matrix
        affines.append(affine)
        # Stop early if specified
        if i > 20 and early_stop:
            break
    # Return images and affines
    if getAffines:
        return images, affines
    else:
        return images, None

# Dataset Class
class SegmentationData(Dataset):
    """
    A custom dataset class for loading and handling segmentation data.
    """
    def __init__(self, image_dir, label_dir, norm_image=False, categorical=False, dtype=np.float32):
        """
        Initializes the dataset by loading images and labels from specified directories.

        Parameters:
            image_dir (str): Directory containing the image files.
            label_dir (str): Directory containing the label files.
            norm_image (bool): If True, normalize images to mean=0 and std=1.
            categorical (bool): If True, convert labels to one-hot categorical format.
            dtype (numpy data type): Desired data type for images and labels.
        """
        self.image_filenames = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                       if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.label_filenames = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir)
                                       if f.endswith('.nii') or f.endswith('.nii.gz')])
        print(f"Image Dir {image_dir}: {len(self.image_filenames)} images")
        print(f"Label Dir {label_dir}: {len(self.label_filenames)} labels")

        # Collect all unique labels to determine the total number of classes
        self.num_classes = self._determine_num_classes()
        print(f"Total number of classes in the dataset: {self.num_classes}")

        self.images, _ = load_data_2D(self.image_filenames, normImage=norm_image, categorical=False, dtype=dtype)
        # For labels, set categorical=True to get one-hot encoding
        self.labels, _ = load_data_2D(self.label_filenames, normImage=False, categorical=categorical,
                                      dtype=dtype, num_classes=self.num_classes)
        if not categorical:
            self.labels = (self.labels > 0).astype(dtype)

        # Initialize segmentation classes mapping
        self.segmentation_classes = None

    def _determine_num_classes(self):
        """
        Determines the total number of unique classes across all label images.

        Returns:
            int: Total number of classes.
        """
        unique_labels = set()
        print("Determining total number of classes in the dataset...")
        for label_name in tqdm(self.label_filenames, desc="Scanning labels"):
            label_img = nib.load(label_name).get_fdata(caching='unchanged')
            if len(label_img.shape) == 3:
                label_img = label_img[:, :, 0]
            unique_labels.update(np.unique(label_img).astype(int).tolist())
        num_classes = max(unique_labels) + 1  # Assuming classes are labeled from 0 to N
        return num_classes

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Parameters:
            idx (int): Index of the data to retrieve.

        Returns:
            tuple: A tuple containing the image and label tensors.
        """
        image = self.images[idx]
        label = self.labels[idx]
        # Add channel dimension if missing
        if len(image.shape) == 2:
            image = image[np.newaxis, :]
        if len(label.shape) == 2:
            label = label[np.newaxis, :]
        elif len(label.shape) == 3 and self.num_classes > 1:
            label = label.transpose(2, 0, 1)  # Move channels to first dimension
        return image, label

    def plot_img_and_labels(self, idx):
        """
        Plots the image and its labels side by side.

        Parameters:
            idx (int): Index of the data to plot.
        """
        image, label = self[idx]
        # Squeeze singleton dimensions for plotting
        image = np.squeeze(image)
        num_labels = label.shape[0]
        height, width = image.shape[-2:]
        print(f"Image Resolution: {width}x{height}")
        # Get the labels present in the image
        if num_labels > 1:
            label_indices = label.argmax(axis=0)
            unique_labels = np.unique(label_indices)
        else:
            unique_labels = np.unique(label)
        print(f"Number of Labels found in image: {len(unique_labels)}")
        print(f"Labels found: {unique_labels}")
        # Set up the plotting grid
        fig, axes = plt.subplots(1, num_labels + 1, figsize=(5 * (num_labels + 1), 5))
        # Plot the original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Image")
        axes[0].axis('off')
        # Plot each label
        for i in range(num_labels):
            label_i = label[i]
            axes[i + 1].imshow(label_i, cmap='gray')
            axes[i + 1].set_title(f"Label {i}")
            axes[i + 1].axis('off')
        plt.tight_layout()
        plt.show()

    def collect_segmentation_classes(self):
        """
        Analyzes the dataset to find all different segmentation classes present
        in the labels, and stores lists of image indices associated with each class.
        """
        self.segmentation_classes = defaultdict(list)

        # Iterate over all images in the dataset
        print("Collecting segmentation classes...")
        for idx in tqdm(range(len(self)), desc="Analyzing labels"):
            label = self.labels[idx]
            # Handle categorical (one-hot encoded) labels
            if self.num_classes > 1:
                # label shape: (num_classes, H, W)
                for class_idx in range(self.num_classes):
                    # Check if class_idx is present in this label
                    if label[class_idx].any():
                        self.segmentation_classes[class_idx].append(idx)
            else:
                # Labels are not one-hot encoded (binary segmentation)
                unique_classes = np.unique(label)
                for class_idx in unique_classes:
                    class_idx = int(class_idx)
                    self.segmentation_classes[class_idx].append(idx)

        # Convert defaultdict to regular dict
        self.segmentation_classes = dict(self.segmentation_classes)
        print("Segmentation classes collected:")
        for class_idx, indices in self.segmentation_classes.items():
            print(f"Class {class_idx}: {len(indices)} images")

    def get_segmentation_classes(self):
        """
        Returns the mapping of segmentation classes to image indices.

        Returns:
            dict: Dictionary with class labels as keys and lists of image indices as values.
        """
        if self.segmentation_classes is None:
            self.collect_segmentation_classes()
        return self.segmentation_classes
