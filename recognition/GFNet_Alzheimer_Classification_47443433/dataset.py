import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, subdir in enumerate(['NC', 'AD']):  # Normal Control and Alzheimer's Disease folders
            folder_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(folder_path):
                print(f"Warning: Directory '{folder_path}' does not exist.")
                continue
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, filename))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations for training and test data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),              # Convert grayscale images to 3 channels
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),      # Slightly resized random crop to add variability
    transforms.RandomRotation(10),                            # Small rotation range to avoid major distortions
    transforms.RandomHorizontalFlip(p=0.5),                   # Horizontal flip with 50% probability
    transforms.ColorJitter(brightness=0.1, contrast=0.1),     # Slight brightness and contrast adjustment
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Small translation to shift position slightly
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Only basic transformations for test data
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),              # Convert grayscale images to 3 channels
    transforms.Resize((224, 224)),                            # Resize to model's input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# Load the dataset
train_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/train', transform=train_transform)
test_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test', transform=test_transform)
