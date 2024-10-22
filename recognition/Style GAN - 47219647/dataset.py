"""
@brief: This file creates a custom dataset and data loader for loading singular class images (e.g. NC, AD) 
from a specified directory. It also applies augmentation such as resizing, grayscaling, and flipping.
@Author: Amlan Nag (s4721964)
"""

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    """
    Custom data set which does not use classes nor files. Hence making it easier
    to load singular classes such as AD individually.
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  
        
        if self.transform:
            image = self.transform(image)

        return image, 0  

def data_set_creator(image_size, batch_size):
    """
    Function that augments the data based on the batch size and image size.
    """
    augmentation_transforms = transforms.Compose([
        # Converts images to greyscale
        transforms.Grayscale(num_output_channels=1),
        # Resizes them based on training stage 
        transforms.Resize((image_size, image_size)),
        # Randomly flips them  
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Locan file directory 
    data_dir = 'recognition/Style GAN - 47219647/AD_NC/train/AD'  
    
    dataset = CustomImageDataset(image_dir=data_dir, transform=augmentation_transforms)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    return data_loader, dataset
