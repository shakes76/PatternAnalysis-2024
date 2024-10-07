import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ADNIDataset(Dataset):
    """
    Args:
        root_dir (string): Directory with all the images
        transform (callable, optional): Optional transform to be applied on a sample
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Loop through the train/test directory
        for class_name in ["NC", "AD"]:
            class_dir = os.path.join(root_dir, class_name)
            label = 0 if class_name == "NC" else 1  # 0 for NC/1 for AD
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        """Return the total number of samples"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image and return it along with the corresponding label"""
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # Load the image using PIL
        image = Image.open(img_path).convert('RGB')
        # Apply transformations 
        if self.transform:
            image = self.transform(image)

        return image, label
    
# Function to create DataLoader for training and testing datasets
def load_data(train_dir, test_dir, batch_size=32):
    """
    Load the train and test datasets using DataLoader
    Args: train_dir (string): Path to the training dataset
        test_dir (string): Path to the testing dataset
        batch_size (int, optional): Number of samples per batch
    Returns: train_loader, test_loader: DataLoader objects for training and testing data
    """
    # Define transformation for the dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and std
                             std=[0.229, 0.224, 0.225])
    ])

    # Create train and test dataset
    train_dataset = ADNIDataset(root_dir=train_dir, transform=transform)
    test_dataset = ADNIDataset(root_dir=test_dir, transform=transform)

    # Create train and test dataset loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader