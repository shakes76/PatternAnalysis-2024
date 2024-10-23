import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np

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
        Loading metadata from the CSV file.
        """
        self.data = pd.read_csv(self.csv_path)

    def split_data(self):
        """
        Split the data into training and testing sets
        """
        train_data, test_data = train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data['target'])
        return train_data, test_data

    def create_dataloaders(self, batch_size=32):
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
        # Statistics
        print(f"Total images: {len(self.data)}")
        print(f"Classes distribution: \n{self.data['target'].value_counts()}")
        print("\nNote: 0 = benign, 1 = malignant")


class SiameseDataset(Dataset):
    """
    Dataset for Siamese Network training and melanoma classification
    """
    def __init__(self, data, img_dir):
        self.data = data
        self.img_dir = img_dir
        self.diagnosis_labels = data['target'].values  # 0 = benign, 1 = malignant
        self.image_ids = data['isic_id'].values

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        # Get the first image and associated diagnosis
        img1_id = self.image_ids[idx]
        img1_diagnosis = self.diagnosis_labels[idx]
        
        # Random choice of same-diagnosis or different-diagnosis pair
        should_get_same_class = np.random.random() > 0.5
        
        if should_get_same_class:
            # Get another image with same diagnosis (both benign or both malignant)
            same_class_indices = np.where(self.diagnosis_labels == img1_diagnosis)[0]
            second_idx = np.random.choice(same_class_indices)
            while second_idx == idx:  # Avoid picking the same image
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
        img1 = load_image(img1_id, self.img_dir)
        img2 = load_image(img2_id, self.img_dir)
        
        return {
            'img1': img1,
            'img2': img2,
            'similarity_label': similarity_label,
            'diagnosis1': torch.tensor(img1_diagnosis, dtype=torch.float32),
            'diagnosis2': torch.tensor(img2_diagnosis, dtype=torch.float32)
        }

def preprocess_image(image):
    """
    Preprocess image for ResNet50 input
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform(image)

def load_image(image_id, img_dir):
    """
    Takes an image from the ISIC-2020 dataset and preprocesses it for the models
    """
    img_path = f'{img_dir}{image_id}.jpg'
    image = Image.open(img_path).convert('RGB')
    return preprocess_image(image)


if __name__ == "__main__":
    # File paths
    csv_path = 'archive/train-metadata.csv'
    img_dir = 'archive/train-image/image/'

    data_manager = DataManager(csv_path, img_dir)
    data_manager.load_data()
    data_manager.create_dataloaders()
    data_manager.print_statistics()