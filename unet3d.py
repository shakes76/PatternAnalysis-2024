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
import nibabel as nib
"""
Brief:
Segment the (downsampled) Prostate 3D data set (see Appendix for link) with the 3D Improved UNet3D [3] 
with all labels having a minimum Dice similarity coefficient of 0.7 on the test set. See also CAN3D [4] 
for more details and use the data augmentation library here for TF or use the appropriate transforms in PyTorch. 
You may begin with the original 3D UNet [5]. You will need to load Nifti file format and sample code is provided in Appendix B. 
[Normal Difficulty- 3D UNet] [Hard Difficulty- 3D Improved UNet]

"""

class CustomDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
        self.labels = labels  
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        label_path = os.path.join(self.img_dir, self.labels[idx])  

        # Load Nifti images
        image = nib.load(img_path).get_fdata()  
        label = nib.load(label_path).get_fdata()  

        #  apply transformations 
        if self.transform:
            image = self.transform(image)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) 
        label = torch.tensor(label, dtype=torch.long) 

        return image, label

# Function to split the dataset into train and test
def split_dataset(img_dir, labels, test_size=0.2, random_state=42):
    # List all Nifti filenames in the directory
    image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    # Split the filenames and labels into train and test sets
    train_filenames, test_filenames, train_labels, test_labels = train_test_split(
        image_filenames, labels, test_size=test_size, random_state=random_state
    )

    return train_filenames, test_filenames, train_labels, test_labels

def main():
    img_dir = "ass\comp3710-ass1\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_MRs_anon"
    labels_dir = "ass\comp3710-ass1\Labelled_weekly_MR_images_of_the_male_pelvis-Xken7gkM-\data\HipMRI_study_complete_release_v1\semantic_labels_anon"
      
    image_filenames = [f for f in os.listdir(img_dir) if f.endswith('.nii.gz')]
    
    train_filenames, test_filenames, train_labels, test_labels = split_dataset(img_dir, labels_dir)

    dataset = CustomDataset(image_filenames, labels=None, img_dir=img_dir, labels_dir=labels_dir, load_3d=False) #! check if 2d or 3d
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)