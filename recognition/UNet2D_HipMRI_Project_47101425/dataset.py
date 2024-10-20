import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import load_data_2D, load_data_3D

# Replace with your actual image directory path
image_dir = "./HipMRI_study_keras_slices_data/keras_slices_seg_test"


class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, normImage=False, categorical=False, dtype=np.float32, 
                 getAffines=False, early_stop=False, load_type='2D', orient=False, transform=None):
        '''
        Initialize the dataset loader with the directory containing the NIFTI files.
        '''
        self.image_dir = image_dir
        self.label_dir = label_dir  # Add a separate label directory
        self.normImage = normImage
        self.categorical = categorical
        self.dtype = dtype
        self.getAffines = getAffines
        self.early_stop = early_stop
        self.load_type = load_type
        self.orient = orient
        self.transform = transform  # Allow optional transformation

        # Load images and labels once during initialization
        self.images = self.load_images()
        self.labels = self.load_labels()

    def load_images(self):
        '''
        Load all images from the directory and return them as a 4D or 5D numpy array.
        '''
        image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) 
                       if f.endswith('.nii') or f.endswith('.nii.gz')]

        if self.load_type == '2D':
            return load_data_2D(image_paths, normImage=self.normImage, 
                                categorical=self.categorical, dtype=self.dtype, 
                                getAffines=self.getAffines, early_stop=self.early_stop)
        elif self.load_type == '3D':
            return load_data_3D(image_paths, normImage=self.normImage, 
                                categorical=self.categorical, dtype=self.dtype, 
                                getAffines=self.getAffines, orient=self.orient, 
                                early_stop=self.early_stop)
        else:
            raise ValueError("load_type should be either '2D' or '3D'")

    def load_labels(self):
        '''
        Load all labels from the label directory. Assumes that the labels correspond to the images.
        '''
        label_paths = [os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir) 
                       if f.endswith('.nii') or f.endswith('.nii.gz')]

        if self.load_type == '2D':
            return load_data_2D(label_paths, normImage=False, categorical=True, dtype=np.uint8)
        elif self.load_type == '3D':
            return load_data_3D(label_paths, normImage=False, categorical=True, dtype=np.uint8)
        else:
            raise ValueError("load_type should be either '2D' or '3D'")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Return the image and corresponding label at index `idx`.
        '''
        image = self.images[idx]
        label = self.labels[idx]  # Load the corresponding label

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# Create the dataset
dataset = MedicalImageDataset(image_dir, normImage=True, load_type='2D')

# Create the DataLoader
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# Example of using the DataLoader in a training loop
for batch_idx, (images, labels) in enumerate(train_loader):
    print(f"Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
