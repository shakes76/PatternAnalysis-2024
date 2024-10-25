""" 
Dataset Loaders
Set of functions and classes to load the ADNI dataset.

Author: Kevin Gu
"""
import os
import re
from torchvision.transforms import v2
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

# Mean and std for the ADNI dataset
ADNI_MEAN = 0.1156
ADNI_STD = 0.2199


def extract_patient_id(file_name):
    """
    Helper function to extract the first part of the file name out, 
    i.e., patientID. For example, 23542_290.jpeg will return 23542.

    Parameters:
        file_name: Name of file
    
    Returns: 
        The patientID from the file name
    """
    return re.match(r"([^_]+)_", file_name).group(1)


class ADNIDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Custom Dataset for loading ADNI data with labels.
        
        Args:
            image_paths (list): List of image file paths.
            labels (list): List of corresponding labels (0 for NC, 1 for AD).
            transform (torchvision.transforms): Image transformations.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Return length of the image path list
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Return the indexed image, with transform applied
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def ADNI_split_validation_set(image_paths: list, labels: list):
    """
    Splits the data on a patient level, then on a class level, ensuring equal
    balance of classes.

    Parameters:
        image_paths: A list of paths to all the images
        labels: A list containing the class labels associated with each image
    Returns:
        train_image_paths: A list of paths to images in train set
        train_labels: A list of labels of images in train set
        val_image_paths: A list of paths to images in validation set
        val_labels: A list of labels of images in validation set
    """
     # Extract patient IDs to ensure no leakage
    patient_ids = [extract_patient_id(os.path.basename(path)) for path in image_paths]
    
    # Create a DataFrame to assist in patient-wise splitting
    data = list(zip(image_paths, labels, patient_ids))
    
    # Perform a patient-wise split
    unique_patients = set(patient_ids)
    train_patients, val_patients = train_test_split(list(unique_patients), test_size=0.1, random_state=42)

    df = pd.DataFrame(data, columns=['image_path', 'label', 'patient_id'])

    # Split patients by class (NC and AD) to maintain class balance
    nc_patients = df[df['label'] == 0]['patient_id'].unique()  # Patients with NC
    ad_patients = df[df['label'] == 1]['patient_id'].unique()  # Patients with AD

    # Split patients with stratification to ensure even class distribution
    train_nc_patients, val_nc_patients = train_test_split(nc_patients, test_size=0.1, random_state=42)
    train_ad_patients, val_ad_patients = train_test_split(ad_patients, test_size=0.1, random_state=42)

    # Combine train and validation patients
    train_patients = list(train_nc_patients) + list(train_ad_patients)
    val_patients = list(val_nc_patients) + list(val_ad_patients)

    # Split the dataset into training and validation based on patient IDs
    train_data = df[df['patient_id'].isin(train_patients)]
    val_data = df[df['patient_id'].isin(val_patients)]

    # Separate image paths and labels
    train_image_paths = train_data['image_path'].tolist()
    train_labels = train_data['label'].tolist()
    
    val_image_paths = val_data['image_path'].tolist()
    val_labels = val_data['label'].tolist()

    return (train_image_paths, train_labels), (val_image_paths, val_labels)
    

def load_adni_data(root_dir, batch_size=32, testing=False):
    """
    Load the ADNI dataset and ensure no leakage by splitting on patient level 
    and also ensuring equal class distribution.
    
    Parameters:
        root_dir (str): Root directory containing the 'train' folder with NC and AD subfolders.
        valid_size (float): Fraction of the training set to be used for validation.
        batch_size (int): Batch size for DataLoader.
        
    Returns:
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
    """
    # Initialize lists for storing image paths and labels
    image_paths = []
    labels = []

    directory = 'train' if not testing else 'test'

    # Load images from NC (Normal Control) and AD (Alzheimer's Disease) folders
    classes = {'NC': 0, 'AD': 1}  # 0 for NC, 1 for AD
    for class_name, label in classes.items():
        class_dir = os.path.join(root_dir, directory, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(label)

    if not testing:  # If not testing, need to split into training and validation set

        train_data, val_data = ADNI_split_validation_set(image_paths, labels)

        # Define image transformations
        transform = v2.Compose([
            v2.CenterCrop((224, 224)),  # Resize to fit model input size (256 x 240)
            v2.RandomVerticalFlip(p=0.5),
            v2.ToTensor(),  # Convert image to PyTorch tensor
            v2.GaussianNoise(),
            v2.Normalize(mean=[ADNI_MEAN], std=[ADNI_STD])
        ])

        # Create ADNIDataset instances for training and validation
        train_dataset = ADNIDataset(*train_data, transform=transform)
        val_dataset = ADNIDataset(*val_data, transform=transform)
        # Create DataLoaders for batching and parallel data loading
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        return train_loader, val_loader

    else:  # Testing
        transform = v2.Compose([
            v2.CenterCrop((224, 224)),  # Resize to fit model input size (256 x 240)
            v2.ToTensor(),  # Convert image to PyTorch tensor
            v2.Normalize(mean=[ADNI_MEAN], std=[ADNI_STD]) 
        ])

        test_dataset = ADNIDataset(image_paths, labels, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        return test_loader