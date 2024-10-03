"""
Two different dataset classes are prepared for training and test seperately.

Please make sure you have write permission to the folder where the dataset is located.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from tqdm import tqdm
from collections import defaultdict

class ADNIDataset(Dataset):
    """ADNI dataset loader.

    This dataset class automatically preprocesses brain images by cropping the brain region 
    and resizing them to 210x210 pixels. It supports loading data for two categories: 
    AD and NC.

    The class can split the dataset into training and validation sets based on 'split_ratio'
    and seed is used for reproducibility. Preprocessing steps such as cropping and resizing are applied 
    during the first run, and processed images are saved for future reuse.
    
    Please make sure you have write permission to the folder where the dataset is located.
    """
    def __init__(self, root, split="train", transform=None, val=False, seed=0, split_ratio=0.8, disable_tqdm=False):
        root = os.path.join(root, split)
        self.root = root
        self.ad_dir = os.path.join(root, 'AD')
        self.nc_dir = os.path.join(root, 'NC')
        self.ad_processed_dir = os.path.join(root, 'AD_processed')
        self.nc_processed_dir = os.path.join(root, 'NC_processed')
        self.disable_tqdm = disable_tqdm
        
        self.preprocess_images()
        
        self.ad_images = [os.path.join(self.ad_processed_dir, f) for f in os.listdir(self.ad_processed_dir)]
        self.nc_images = [os.path.join(self.nc_processed_dir, f) for f in os.listdir(self.nc_processed_dir)]
        
        self.images = self.ad_images + self.nc_images
        self.labels = [1] * len(self.ad_images) + [0] * len(self.nc_images)
        
        self.transform = transform
        self.val = val
        self.seed = seed
        self.split_ratio = split_ratio
        
        self.mask = self._generate_mask()
    
    def _generate_mask(self):
        random.seed(self.seed)
        if self.split_ratio == 1:
            return [True] * len(self.images)
        mask = [random.random() < self.split_ratio for _ in range(len(self.images))]
        return mask if not self.val else [not m for m in mask]

    def preprocess_images(self):
        if not os.path.exists(self.ad_processed_dir):
            print("Processing AD")
            os.makedirs(self.ad_processed_dir)
            self._process_directory(self.ad_dir, self.ad_processed_dir)
        
        if not os.path.exists(self.nc_processed_dir):
            print("Processing NC")
            os.makedirs(self.nc_processed_dir)
            self._process_directory(self.nc_dir, self.nc_processed_dir)

    def _process_directory(self, input_dir, output_dir):
        for filename in tqdm(os.listdir(input_dir), disable=self.disable_tqdm):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                self._process_image(input_path, output_path)

    def _process_image(self, input_path, output_path):
        # Read image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Crop the relevant region
        cropped = self._crop_brain_region(image)
        
        h, w = cropped.shape
    
        # Calculate scaling factor to make the longer side 210 pixels
        scale = 210 / max(h, w)
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize the image
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate padding
        top = (210 - new_h) // 2
        bottom = 210 - new_h - top
        left = (210 - new_w) // 2
        right = 210 - new_w - left
        
        # Add padding
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=0)
        
        # Save the processed image
        cv2.imwrite(output_path, padded)

    def _crop_brain_region(self, image):
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the coordinates of non-zero pixels
        coords = cv2.findNonZero(binary)
        
        # Get the smallest rectangle that encloses all non-zero pixels
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image
        return image[y:y+h, x:x+w]

    def __len__(self):
        return sum(self.mask)

    def __getitem__(self, idx):
        real_idx = [i for i, m in enumerate(self.mask) if m][idx]
        img_path = self.images[real_idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[real_idx]
        return image, label


class ADNIDatasetTest(Dataset):
    """ADNI Test dataset loader.

    This dataset class automatically preprocesses brain images by cropping the brain region 
    and resizing them to 210x210 pixels. It supports loading test data, where each group of images
    is identified by the same leading number in the filename. For each group, this class returns 
    a stack of 20 images ordered by the second number in descending order, with labels 1 for AD 
    and 0 for NC. The processed images are read from the corresponding 'AD_processed' and 'NC_processed' folders.

    If the processed images are not already available, the class will process and save them in these folders
    during the first run.

    Please make sure you have write permission to the folder where the dataset is located.
    """
    def __init__(self, root, transform=None, disable_tqdm=False):
        self.root = os.path.join(root, 'test')
        self.ad_dir = os.path.join(self.root, 'AD')
        self.nc_dir = os.path.join(self.root, 'NC')
        self.ad_processed_dir = os.path.join(self.root, 'AD_processed')
        self.nc_processed_dir = os.path.join(self.root, 'NC_processed')
        self.disable_tqdm = disable_tqdm
        self.transform = transform
        
        # Check if images are already processed, if not, process them
        self.preprocess_images()

        # Group images based on the leading number and assign labels
        self.ad_image_groups = self._group_images(self.ad_processed_dir, label=1)
        self.nc_image_groups = self._group_images(self.nc_processed_dir, label=0)

        # Combine AD and NC groups
        self.image_groups = self.ad_image_groups + self.nc_image_groups

    def preprocess_images(self):
        """Check if the processed folders exist, and if not, process the images."""
        if not os.path.exists(self.ad_processed_dir):
            print("Processing AD images...")
            os.makedirs(self.ad_processed_dir)
            self._process_directory(self.ad_dir, self.ad_processed_dir)
        
        if not os.path.exists(self.nc_processed_dir):
            print("Processing NC images...")
            os.makedirs(self.nc_processed_dir)
            self._process_directory(self.nc_dir, self.nc_processed_dir)

    def _process_directory(self, input_dir, output_dir):
        """Process and save images in the given directory."""
        for filename in tqdm(os.listdir(input_dir), disable=self.disable_tqdm):
            if filename.lower().endswith('.jpeg'):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                self._process_image(input_path, output_path)

    def _process_image(self, input_path, output_path):
        """Preprocess image by cropping, resizing, and padding, then save it."""
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Crop the relevant region
        cropped = self._crop_brain_region(image)
        
        h, w = cropped.shape
    
        # Calculate scaling factor to make the longer side 210 pixels
        scale = 210 / max(h, w)
        
        # Calculate new dimensions
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize the image
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Calculate padding
        top = (210 - new_h) // 2
        bottom = 210 - new_h - top
        left = (210 - new_w) // 2
        right = 210 - new_w - left
        
        # Add padding
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                    cv2.BORDER_CONSTANT, value=0)
        
        # Save the processed image
        cv2.imwrite(output_path, padded)

    def _crop_brain_region(self, image):
        """Crop the brain region using Otsu's thresholding."""
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the coordinates of non-zero pixels
        coords = cv2.findNonZero(binary)
        
        # Get the smallest rectangle that encloses all non-zero pixels
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop the image
        return image[y:y+h, x:x+w]

    def _group_images(self, processed_dir, label):
        """Group images based on the leading number in the filename and assign labels."""
        image_groups = []
        group_dict = defaultdict(list)
        
        # Iterate through processed directory and group images
        for filename in os.listdir(processed_dir):
            if filename.lower().endswith('.jpeg'):
                group_number = filename.split('_')[0]
                group_dict[group_number].append(filename)
        
        # Sort images within each group by the second number in descending order
        for group_number, filenames in group_dict.items():
            filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
            
            # Check that each group has exactly 20 images, otherwise raise an error
            if len(filenames) != 20:
                raise ValueError(f"Group {group_number} does not contain exactly 20 images.")
            
            # Append group information with label
            image_groups.append({
                'group_number': group_number,
                'filenames': filenames,
                'label': label
            })
        
        return image_groups

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        """Return a 20-channel image stack for the group along with the label."""
        # Get group info based on idx
        group_info = self.image_groups[idx]
        group_number = group_info['group_number']
        image_filenames = group_info['filenames']
        label = group_info['label']
        
        # Initialize a list to store images
        image_stack = []
        
        # Determine the appropriate processed directory based on label
        processed_dir = self.ad_processed_dir if label == 1 else self.nc_processed_dir
        
        for filename in image_filenames:
            # Construct the full path of the image
            image_path = os.path.join(processed_dir, filename)
            
            # Preprocess the image (crop, resize, and pad)
            image = Image.open(image_path).convert('L')
            
            # Apply any given transformations
            if self.transform:
                image = self.transform(image)
            
            image_stack.append(image)
        
        # Stack the 20 images into a single tensor of shape (20, 1, 210, 210)
        image_stack = np.stack(image_stack, axis=0)
        
        return torch.tensor(image_stack, dtype=torch.float32), torch.tensor(label).float()