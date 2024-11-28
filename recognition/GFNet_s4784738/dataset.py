"""
Creates data loaders for brain images from the ADNI dataset.

Benjamin Thatcher 
s4784738    
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms

class ADNIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialises the image dataset using ADNI brain data.

        root_dir: Directory containing 'AD' and 'ND' subfolders with brain images inside them
        transform: Optional transform to be applied to an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Traverse the directory to gather image paths and labels
        for label_name in ['AD', 'NC']:
            label_dir = os.path.join(root_dir, label_name)
            # Set the label to 1 for Alzheimer's (AD) and 0 for Normal (NC)
            label = 1 if label_name == 'AD' else 0
            
            for image_name in sorted(os.listdir(label_dir)):
                # Ensure the images are all valid (all ADNI images should be .jpeg)
                if image_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(label)

    def __len__(self):
        """
        Returns the length of the image dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Gets the image and label and a certain index in the dataset

        idx: The index in the dataset of the image to be retrieved
        Returns: The image (in grayscale) and its label
        """
        image_path = self.image_paths[idx]
        # Open the image as grayscale
        image = Image.open(image_path).convert('L')
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        # Transform the image if transforms are being used
        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loader(root_dir, dataset='train', batch_size=32, shuffle=True, split=0.2, num_workers=0):
    """
    Gets a dataloader with characteristics specified by the user.

    root_dir: The root directory in which ADNI images can be found
    dataset: Specifies if the dataloader is for a training/validation or testing split
    batch_size: The batch size for images 
    shuffle: True is images should be shuffled, and False otherwise
    split: The split between training and validation datasets
    num_workers: The number of workers
    """
    # Set up different transforms for training/validating and testing splits
    if dataset == 'train':
        transform = transforms.Compose([
            # Resize to 224x224 from the default size of 256x240 pixels
            transforms.Resize((224, 224)),
            transforms.Grayscale(),

            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=10, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.15, 3)),

            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    else:
        transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    adni_dataset = ADNIDataset(root_dir=root_dir, transform=transform)
    
    if dataset == 'train':
        # Split the train dataset into training and validation sets
        train_size = int((1 - split) * len(adni_dataset))
        val_size = len(adni_dataset) - train_size
        train_dataset, val_dataset = random_split(adni_dataset, [train_size, val_size])

        # Return the training and validation data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
        return train_loader, val_loader
    else:
        # The testing dataset
        data_loader = DataLoader(adni_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return data_loader
