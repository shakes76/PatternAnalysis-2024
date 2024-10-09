import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:   root_dir (string): Directory with all the images
                transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # Loop through the directory to get image paths (both AD and NC images)
        for class_name in ["NC", "AD"]:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.image_paths.append(img_path)

    def __len__(self):
        """Return the total number of samples"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image and return it"""
        img_path = self.image_paths[idx]

        # Load the image 
        image = Image.open(img_path).convert('RGB') 

        # Apply transformations 
        if self.transform:
            image = self.transform(image)

        return image 

# Function to create DataLoader for training the GAN
def load_data(train_dir, test_dir, batch_size=32):
    """
    Load the training dataset using DataLoader
    Args:   train_dir (string): Path to the training dataset
            batch_size (int, optional): Number of samples per batch
    Returns:
            train_loader: DataLoader object for training data
    """
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to 128x128
        transforms.ToTensor(),  # Convert to tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Create the dataset and data loaders
    train_dataset = ADNIDataset(root_dir=train_dir, transform=transform)
    test_dataset = ADNIDataset(root_dir=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) 

    return train_loader, test_loader

# if __name__ == '__main__':
#     train_dir = r'C:/Users/Admin/Downloads/ADNI_AD_NC_2D/AD_NC/train'
#     test_dir = r'C:/Users/Admin/Downloads/ADNI_AD_NC_2D/AD_NC/test'
#     train_loader, test_loader = load_data(train_dir, test_dir)

#     for images in train_loader:
#         print(f"Training Batch size: {images.size()}")
#         break

#     for images in test_loader:
#         print(f"Testing Batch size: {images.size()}")
#         break 