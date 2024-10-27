"""
- Manages and preprocesses melanoma dataset for model training
- Includes data loading, augmentation, and DataLoader generation

@author: richardchantra
@student_number: 43032053
"""

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np
import argparse

class DataManager:
    """
    Managing the data flows and processing for the ISIC-2020 dataset
    """
    def __init__(self, csv_path, img_dir):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.data = None
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        """
        Loading metadata from the CSV file
        """
        self.data = pd.read_csv(self.csv_path)

    def balance_dataset(self, data, data_augmentation='oversampling', target_ratio=None):
        """
        Balance dataset using specified sampling strategy
        target_ratio is the ratio of benign desired in the balanced dataset
        """
        if data_augmentation is None:
            return data
            
        # Separate majority and minority classes
        malignant = data[data['target'] == 1]
        benign = data[data['target'] == 0]

        if data_augmentation == 'oversampling':
            n_samples = len(benign)
            malignant_oversampled = malignant.sample(n=n_samples, replace=True, random_state=42)
            balanced_data = pd.concat([benign, malignant_oversampled])
        elif data_augmentation == 'undersampling':
            n_samples = len(malignant)
            benign_undersampled = benign.sample(n=n_samples, random_state=42)
            balanced_data = pd.concat([malignant, benign_undersampled])
        elif data_augmentation == 'ratio' and target_ratio is not None:
            # Calculate target numbers for the specified ratio
            total_samples = len(data)
            target_benign_samples = int(total_samples * target_ratio)
            target_malignant_samples = total_samples - target_benign_samples
            
            # Downsample benign or keep to maintain target
            if len(benign) > target_benign_samples:
                benign_sampled = benign.sample(n=target_benign_samples, random_state=42)
            else:
                benign_sampled = benign
                
            # Oversample malignant to reach target
            malignant_oversampled = malignant.sample(n=target_malignant_samples, replace=True, random_state=42)
            balanced_data = pd.concat([benign_sampled, malignant_oversampled])
        else:
            raise ValueError(f"Invalid parameter for: data_augmentation or target_ratio")

        return balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    def split_data(self):
        """
        Split the data into training and testing sets and balance
        """
        # Normal train test split
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data['target'])
        # Balance dataset based on oversampling, undersampling or oversampling using a ratio
        balanced_train_data = self.balance_dataset(train_data, data_augmentation='ratio', target_ratio=0.67)
        
        return balanced_train_data, test_data

    def create_dataloaders(self, batch_size=256):
        """
        Creating torch DataLoader objects for training and testing
        """
        train_data, test_data = self.split_data()
        
        # Create DataLoader
        train_dataset = SiameseDataset(train_data, self.img_dir)
        test_dataset = SiameseDataset(test_data, self.img_dir)

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            pin_memory=True,
            num_workers=4
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=4
        )

    def print_statistics(self):
        """
        Print dataset statistics before and after balancing
        """
        # Original dataset statistics
        class_distribution = self.data['target'].value_counts()
        print(f"Original dataset statistics:\n"
            f"Total images: {len(self.data)}\n"
            f"Classes distribution:\n{class_distribution}\n")

        # Split and display training/testing data statistics
        train_data, test_data = self.split_data()
        train_distribution = train_data['target'].value_counts()
        test_distribution = test_data['target'].value_counts()
        print(f"After balancing training data:\n"
            f"Training set distribution:\n{train_distribution}\n"
            f"Test set distribution:\n{test_distribution}\n"
            f"\nNote: 0 = benign, 1 = malignant")

class SiameseDataset(Dataset):
    """
    Dataset for Siamese Network training and melanoma classification
    """
    def __init__(self, data, img_dir):
        self.data = data
        self.img_dir = img_dir
        self.diagnosis_labels = data['target'].values
        self.image_ids = data['isic_id'].values
        
        # resize and normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
])
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Takes an index and returns a pair of images with their labels
        """
        # Get the first image and diagnosis label
        img1_id = self.image_ids[idx]
        img1_diagnosis = self.diagnosis_labels[idx]
        
        # Random choice of same-diagnosis or different-diagnosis pair
        should_get_same_class = np.random.random() > 0.5
        
        if should_get_same_class:
            # Get another image with same diagnosis label
            same_class_indices = np.where(self.diagnosis_labels == img1_diagnosis)[0]
            second_idx = np.random.choice(same_class_indices)
            while second_idx == idx:  # Don't pick same pair
                second_idx = np.random.choice(same_class_indices)
            img2_id = self.image_ids[second_idx]
            similarity_label = torch.tensor(0.0)  # 0 = similar pair
        else:
            # Get an image with different diagnosis
            other_class_indices = np.where(self.diagnosis_labels != img1_diagnosis)[0]
            second_idx = np.random.choice(other_class_indices)
            img2_id = self.image_ids[second_idx]
            similarity_label = torch.tensor(1.0)  # 1 = dissimilar pair
        
        # Get second image's diagnosis
        img2_diagnosis = self.diagnosis_labels[second_idx]
        
        # Load and preprocess images
        img1 = self.load_and_transform(img1_id)
        img2 = self.load_and_transform(img2_id)
        
        return {
            'img1': img1,
            'img2': img2,
            'similarity_label': similarity_label,
            'diagnosis1': torch.tensor(img1_diagnosis, dtype=torch.float32),
            'diagnosis2': torch.tensor(img2_diagnosis, dtype=torch.float32)
        }

    def load_and_transform(self, image_id, threshold=0.7):
        """
        Load image and apply random augmentations if over theshold
        """
        img_path = f'{self.img_dir}{image_id}.jpg'
        image = Image.open(img_path).convert('RGB')
        
        # 30% chance of augmentation
        if np.random.random() > threshold:
            # List of possible augmentations: equal chance of a flip and a rotation
            augmentations = [
                transforms.functional.hflip,
                transforms.functional.vflip,
                lambda img: transforms.functional.rotate(img, np.random.uniform(0, 360)),
                lambda img: transforms.functional.rotate(img, np.random.uniform(0, 360))
            ]
            
            # Random augmentation of image
            aug_func = np.random.choice(augmentations)
            image = aug_func(image)
        
        # Apply standard preprocessing
        return self.transform(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Manager for Melanoma Classification")
    parser.add_argument('--csv_path', type=str, default='archive/train-metadata.csv',
                        help='Path to the CSV metadata file')
    parser.add_argument('--img_dir', type=str, default='archive/train-image/image/',
                        help='Directory path to the image files')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for DataLoader')
    args = parser.parse_args()

    # Initialize DataManager with arguments
    data_manager = DataManager(args.csv_path, args.img_dir)
    data_manager.load_data()
    data_manager.create_dataloaders(batch_size=args.batch_size)
    data_manager.print_statistics()