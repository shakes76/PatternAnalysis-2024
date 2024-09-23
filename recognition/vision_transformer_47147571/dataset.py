import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import shutil

class ADNIDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.ad_dir = os.path.join(root, 'AD')
        self.nc_dir = os.path.join(root, 'NC')
        self.ad_processed_dir = os.path.join(root, 'AD_processed')
        self.nc_processed_dir = os.path.join(root, 'NC_processed')
        
        self.preprocess_images()
        
        self.ad_images = [os.path.join(self.ad_processed_dir, f) for f in os.listdir(self.ad_processed_dir)]
        self.nc_images = [os.path.join(self.nc_processed_dir, f) for f in os.listdir(self.nc_processed_dir)]
        
        self.images = self.ad_images + self.nc_images
        self.labels = [1] * len(self.ad_images) + [0] * len(self.nc_images)
        
        self.transform = transform

    def preprocess_images(self):
        if not os.path.exists(self.ad_processed_dir):
            os.makedirs(self.ad_processed_dir)
            self._process_directory(self.ad_dir, self.ad_processed_dir)
        
        if not os.path.exists(self.nc_processed_dir):
            os.makedirs(self.nc_processed_dir)
            self._process_directory(self.nc_dir, self.nc_processed_dir)

    def _process_directory(self, input_dir, output_dir):
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                self._process_image(input_path, output_path)

    def _process_image(self, input_path, output_path):
        # Read image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        # Crop the relevant region
        cropped = self._crop_brain_region(image)
        
        # Resize to 210x210
        resized = cv2.resize(cropped, (210, 210), interpolation=cv2.INTER_CUBIC)
        
        # Save the processed image
        cv2.imwrite(output_path, resized)

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
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convert image to grayscale
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

