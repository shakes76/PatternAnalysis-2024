import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import numpy as np

# File paths
csv_path = 'archive/train-metadata.csv'
img_dir = 'archive/train-image/image/'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load metadata
data = pd.read_csv(csv_path)

# Statistics
print(f"Total images: {len(data)}")
print(f"Classes distribution: \n{data['target'].value_counts()}")

# Head of metadata
print("\nFirst few rows of metadata:")
print(data.head())

# Define preprocessing transform for ResNet50
def preprocess_image(image):
    """Preprocess image for ResNet50 input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform(image)

# Open and preprocess image
def load_image(image_id):
    img_path = f'{img_dir}{image_id}.jpg'
    image = Image.open(img_path).convert('RGB')  # Ensure RGB format
    return preprocess_image(image)

# Train test split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['target'])

# Size of splits
print(f"\nTraining set size: {len(train_data)}")
print(f"Testing set size: {len(test_data)}")

# Dataset class for Siamese Network
class SiameseDataset(Dataset):
    def __init__(self, data, img_dir):
        self.data = data
        self.img_dir = img_dir
        self.labels = data['target'].values
        self.image_ids = data['isic_id'].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve image
        img1_id = self.image_ids[idx]
        img1_label = self.labels[idx]
        
        # Randomly choice for: similar or dissimilar pair
        should_get_same_class = np.random.random() > 0.5
        
        if should_get_same_class:
            # Second image from the same class
            same_class_indices = np.where(self.labels == img1_label)[0]
            second_idx = np.random.choice(same_class_indices)
            while second_idx == idx:  # Don't pick the same image
                second_idx = np.random.choice(same_class_indices)
            img2_id = self.image_ids[second_idx]
            pair_label = torch.tensor(0.0)
        else:
            # Second image from the other class
            other_class_indices = np.where(self.labels != img1_label)[0]
            second_idx = np.random.choice(other_class_indices)
            img2_id = self.image_ids[second_idx]
            pair_label = torch.tensor(1.0)
        
        # Load and preprocess both images
        img1 = load_image(img1_id)
        img2 = load_image(img2_id)
        
        return img1, img2, pair_label

# Create DataLoader
def get_dataloaders(batch_size=32):
    train_dataset = SiameseDataset(train_data, img_dir)
    test_dataset = SiameseDataset(test_data, img_dir)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True
    )
    
    return train_loader, test_loader

# Create loaders
train_loader, test_loader = get_dataloaders(batch_size=32)

# Test loading
if __name__ == "__main__":
    # Retrieving a batch from dataloader
    batch = next(iter(train_loader))
    sample_img1, sample_img2, sample_label = batch
    
    print("\nSample batch data:")
    print("Batch size:", sample_img1.shape[0])
    print("Image 1 shape:", sample_img1.shape)
    print("Image 2 shape:", sample_img2.shape)
    print("Labels shape:", sample_label.shape)
    print("Sample labels:", sample_label[:5].tolist())