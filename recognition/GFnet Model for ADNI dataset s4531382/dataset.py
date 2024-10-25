import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from tqdm import tqdm
from collections import defaultdict

class ADNITrainDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing ADNI training images.

    This class handles automatic preprocessing of brain images by cropping the brain region
    and resizing them to 210x210 pixels. It supports two categories:
    - AD (Alzheimer's Disease)
    - NC (Normal Control)

    The dataset can be split into training and validation sets based on 'split_ratio'.
    Preprocessing steps like cropping and resizing are applied during the first run,
    and the processed images are saved for future use.
    """

    def __init__(self, data_dir, mode="train", transform=None, validation=False,
                 random_seed=0, split_ratio=0.9, disable_progress=False):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Root directory of the dataset.
            mode (str): Dataset mode, either 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            validation (bool): If True, create a validation set.
            random_seed (int): Seed for random number generator.
            split_ratio (float): Proportion of the dataset to include in the training split.
            disable_progress (bool): If True, disable the progress bar.
        """
        self.root_dir = os.path.join(data_dir, mode)
        self.ad_dir = os.path.join(self.root_dir, 'AD')
        self.nc_dir = os.path.join(self.root_dir, 'NC')
        self.ad_processed_dir = os.path.join(self.root_dir, 'AD_processed')
        self.nc_processed_dir = os.path.join(self.root_dir, 'NC_processed')
        self.transform = transform
        self.validation = validation
        self.random_seed = random_seed
        self.split_ratio = split_ratio
        self.disable_progress = disable_progress

        # Preprocess images if not already done
        self._preprocess_images()

        # Load image paths and labels
        self.ad_images = [os.path.join(self.ad_processed_dir, f) for f in os.listdir(self.ad_processed_dir)]
        self.nc_images = [os.path.join(self.nc_processed_dir, f) for f in os.listdir(self.nc_processed_dir)]
        self.all_images = self.ad_images + self.nc_images
        self.labels = [1] * len(self.ad_images) + [0] * len(self.nc_images)

        # Generate train/validation mask
        self.mask = self._create_mask()

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return sum(self.mask)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (sample image, label)
        """
        # Get the indices of the samples included in the current split
        real_idx = [i for i, m in enumerate(self.mask) if m][idx]

        # Load the image
        img_path = self.all_images[real_idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Get the label
        label = self.labels[real_idx]
        return image, label

    def _create_mask(self):
        """
        Create a mask to split the dataset into training and validation sets.

        Returns:
            list: A list of booleans indicating whether each sample is included in the split.
        """
        random.seed(self.random_seed)
        if self.split_ratio == 1:
            # If split_ratio is 1, include all samples
            return [True] * len(self.all_images)
        else:
            # Randomly include samples based on split_ratio
            mask = [random.random() < self.split_ratio for _ in range(len(self.all_images))]
            # If validation set is requested, invert the mask
            if self.validation:
                mask = [not m for m in mask]
            return mask

    def _preprocess_images(self):
        """
        Preprocess images by cropping and resizing, and save the processed images.

        This method checks if the processed directories exist. If not, it processes
        the images and saves them to the processed directories.
        """
        # Process AD images
        if not os.path.exists(self.ad_processed_dir):
            print("Processing AD images...")
            os.makedirs(self.ad_processed_dir)
            self._process_directory(self.ad_dir, self.ad_processed_dir)

        # Process NC images
        if not os.path.exists(self.nc_processed_dir):
            print("Processing NC images...")
            os.makedirs(self.nc_processed_dir)
            self._process_directory(self.nc_dir, self.nc_processed_dir)

    def _process_directory(self, source_dir, target_dir):
        """
        Process all images in a directory.

        Args:
            source_dir (str): Directory containing the original images.
            target_dir (str): Directory to save the processed images.
        """
        # Iterate over all files in the source directory
        for filename in tqdm(os.listdir(source_dir), disable=self.disable_progress):
            # Process only image files
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(source_dir, filename)
                tgt_path = os.path.join(target_dir, filename)
                self._process_single_image(src_path, tgt_path)

    def _process_single_image(self, input_path, output_path):
        """
        Process a single image: crop, resize, pad, and save.

        Args:
            input_path (str): Path to the original image.
            output_path (str): Path to save the processed image.
        """
        # Load image in grayscale
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Crop the brain region
        cropped_image = self._crop_brain(image)

        # Get original dimensions
        h, w = cropped_image.shape

        # Calculate scaling factor to resize the largest dimension to 210 pixels
        scale_factor = 210 / max(h, w)

        # Compute new dimensions
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize the image using Lanczos interpolation
        resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Calculate padding to make the image 210x210 pixels
        pad_top = (210 - new_h) // 2
        pad_bottom = 210 - new_h - pad_top
        pad_left = (210 - new_w) // 2
        pad_right = 210 - new_w - pad_left

        # Pad the image with zeros (black pixels)
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=0)

        # Save the processed image
        cv2.imwrite(output_path, padded_image)

    def _crop_brain(self, image):
        """
        Crop the brain region from the image by removing the background.

        Args:
            image (numpy.ndarray): Grayscale image array.

        Returns:
            numpy.ndarray: Cropped image containing only the brain region.
        """
        # Apply Otsu's thresholding to create a binary mask
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find coordinates of the non-zero regions in the mask
        coordinates = cv2.findNonZero(binary_mask)

        # Compute the bounding rectangle of the brain region
        x, y, w, h = cv2.boundingRect(coordinates)

        # Crop and return the brain region
        return image[y:y + h, x:x + w]


class ADNITestDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing ADNI test images.

    This class handles automatic preprocessing of brain images by cropping the brain region
    and resizing them to 210x210 pixels. It returns individual images and their labels.
    """

    def __init__(self, data_dir, transform=None, disable_progress=False):
        """
        Initialize the dataset.

        Args:
            data_dir (str): Root directory of the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            disable_progress (bool): If True, disable the progress bar.
        """
        self.root_dir = os.path.join(data_dir, 'test')
        self.ad_dir = os.path.join(self.root_dir, 'AD')
        self.nc_dir = os.path.join(self.root_dir, 'NC')
        self.ad_processed_dir = os.path.join(self.root_dir, 'AD_processed')
        self.nc_processed_dir = os.path.join(self.root_dir, 'NC_processed')
        self.transform = transform
        self.disable_progress = disable_progress

        # Preprocess images if not already done
        self._preprocess_images()

        # Load image paths and labels
        self.image_groups = []
        self.image_groups.extend(self._group_images(self.ad_processed_dir, label=1))
        self.image_groups.extend(self._group_images(self.nc_processed_dir, label=0))

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.image_groups)

    def __getitem__(self, idx):
        # Load the image and label
        img_path = self.image_groups[idx]
        group_num = img_path['group_number']
        sorted_filenames = img_path['filenames']
        label = img_path['label']

        # Initialize a list to store images
        image_stack = []
        
        # Determine the appropriate processed directory based on label
        processed_dir = self.ad_processed_dir if label == 1 else self.nc_processed_dir
        
        for filename in sorted_filenames:
            # load data
            image_path = os.path.join(processed_dir, filename)
            
            image = Image.open(image_path).convert('L')
            
            # Apply any given transformations
            if self.transform:
                image = self.transform(image)
            
            image_stack.append(image)
        
        # Stack the 20 images into a single tensor of shape (20, 1, 210, 210)
        image_stack = np.stack(image_stack, axis=0)
        
        return torch.tensor(image_stack, dtype=torch.float32), torch.tensor(label).float()

    def _preprocess_images(self):
        """
        Preprocess images by cropping and resizing, and save the processed images.

        This method checks if the processed directories exist. If not, it processes
        the images and saves them to the processed directories.
        """
        # Process AD images
        if not os.path.exists(self.ad_processed_dir):
            print("Processing AD test images...")
            os.makedirs(self.ad_processed_dir)
            self._process_directory(self.ad_dir, self.ad_processed_dir)

        # Process NC images
        if not os.path.exists(self.nc_processed_dir):
            print("Processing NC test images...")
            os.makedirs(self.nc_processed_dir)
            self._process_directory(self.nc_dir, self.nc_processed_dir)

    def _process_directory(self, source_dir, target_dir):
        """
        Process all images in a directory.

        Args:
            source_dir (str): Directory containing the original images.
            target_dir (str): Directory to save the processed images.
        """
        # Iterate over all files in the source directory
        for filename in tqdm(os.listdir(source_dir), disable=self.disable_progress):
            # Process only JPEG image files
            if filename.lower().endswith('.jpeg'):
                src_path = os.path.join(source_dir, filename)
                tgt_path = os.path.join(target_dir, filename)
                self._process_single_image(src_path, tgt_path)

    def _process_single_image(self, input_path, output_path):
        """
        Process a single image: crop, resize, pad, and save.

        Args:
            input_path (str): Path to the original image.
            output_path (str): Path to save the processed image.
        """
        # Load image in grayscale
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Crop the brain region
        cropped_image = self._crop_brain(image)

        # Get original dimensions
        h, w = cropped_image.shape

        # Calculate scaling factor to resize the largest dimension to 210 pixels
        scale_factor = 210 / max(h, w)

        # Compute new dimensions
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize the image using Lanczos interpolation
        resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Calculate padding to make the image 210x210 pixels
        pad_top = (210 - new_h) // 2
        pad_bottom = 210 - new_h - pad_top
        pad_left = (210 - new_w) // 2
        pad_right = 210 - new_w - pad_left

        # Pad the image with zeros (black pixels)
        padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=0)

        # Save the processed image
        cv2.imwrite(output_path, padded_image)

    def _crop_brain(self, image):
        """
        Crop the brain region from the image by removing the background.

        Args:
            image (numpy.ndarray): Grayscale image array.

        Returns:
            numpy.ndarray: Cropped image containing only the brain region.
        """
        # Apply Otsu's thresholding to create a binary mask
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find coordinates of the non-zero regions in the mask
        coordinates = cv2.findNonZero(binary_mask)

        # Compute the bounding rectangle of the brain region
        x, y, w, h = cv2.boundingRect(coordinates)

        # Crop and return the brain region
        return image[y:y + h, x:x + w]


    def _group_images(self, processed_dir, label):
        """
        Group images by leading number in filenames and assign labels.

        Args:
            processed_dir (str): Directory containing the processed images.
            label (int): Label to assign to the groups (1 for AD, 0 for NC).

        Returns:
            list: A list of dictionaries containing group information.
        """
        group_dict = defaultdict(list)
        image_groups = []

        # Iterate over all files in the processed directory
        for filename in os.listdir(processed_dir):
            if filename.lower().endswith('.jpeg'):
                # Extract the group number from the filename (leading number before '_')
                group_num = filename.split('_')[0]
                group_dict[group_num].append(filename)

        # Organize and validate each group
        for group_num, filenames in group_dict.items():
            # Sort filenames based on the second number in descending order
            sorted_filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)

            # Check that each group contains exactly 20 images
            if len(sorted_filenames) != 20:
                raise ValueError(f"Group {group_num} does not contain exactly 20 images.")

            # Add group information to the list
            image_groups.append({
                'group_number': group_num,
                'filenames': sorted_filenames,
                'label': label
            })

        return image_groups
