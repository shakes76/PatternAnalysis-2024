import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
"""
Brief:
Segment the (downsampled) Prostate 3D data set (see Appendix for link) with the 3D Improved UNet3D [3] 
with all labels having a minimum Dice similarity coefficient of 0.7 on the test set. See also CAN3D [4] 
for more details and use the data augmentation library here for TF or use the appropriate transforms in PyTorch. 
You may begin with the original 3D UNet [5]. You will need to load Nifti file format and sample code is provided in Appendix B. 
[Normal Difficulty- 3D UNet] [Hard Difficulty- 3D Improved UNet]

"""

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = labels  
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = self.labels[idx]  # Get the corresponding label

        if self.transform:
            image = self.transform(image)

        return image, label  # Return both image and label

# Transforms for data augmentation and normalization
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalization values for grayscale
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalization values for grayscale
])

# Function to split the dataset into train and test
def split_dataset(img_dir, labels, test_size=0.2, random_state=42):
    # List all image filenames in the directory
    image_filenames = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Split the filenames and labels into train and test sets
    train_filenames, test_filenames, train_labels, test_labels = train_test_split(
        image_filenames, labels, test_size=test_size, random_state=random_state
    )

    return train_filenames, test_filenames, train_labels, test_labels

train_filenames, test_filenames, train_labels, test_labels = split_dataset("ass\comp3710-ass1\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon", 
                                                                        "ass\comp3710-ass1\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon")

# Create datasets for training and testing
train_dataset = CustomImageDataset(train_filenames, train_labels, "ass\comp3710-ass1\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon")
test_dataset = CustomImageDataset(test_filenames, test_labels, "ass\comp3710-ass1\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon")