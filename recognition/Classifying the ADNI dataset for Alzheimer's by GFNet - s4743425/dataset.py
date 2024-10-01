"""
For loading the ADNI dataset and pre processing the data
"""

import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
import numpy as np
import torchvision
from PIL import Image

# Class to load and process the images
class ADNIDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        # Use the transform specific to the mode (train, test, val)
        self.transform = transform[mode]  

        # assumes that the dataset is structured in subfolders (AD and NC)
        self.image_filenames = []
        for label_dir in os.listdir(self.data_dir):
            label_path = os.path.join(self.data_dir, label_dir)
            if os.path.isdir(label_path):
                for file_name in os.listdir(label_path):
                    if file_name.endswith(('.jpeg')):
                        self.image_filenames.append((os.path.join(label_dir, file_name), label_dir))

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name, label = self.image_filenames[idx]
        img_path = os.path.join(self.data_dir, img_name)

        # Open the image and check all same
        image = Image.open(img_path).convert("RGB")

        # Apply the transformation based on the mode (train, test, val)
        if self.transform:
            image = self.transform(image)

        # Map label to an index (AD -> 0, NC -> 1)
        label_idx = 0 if label == 'AD' else 1
        
        return image, label_idx


if __name__ == "__main__":

    # The path when running locally
    data_directory = '../../../AD_NC'
    #the path to the directory on Rangpur
    #data_directory = '/home/groups/comp3710/ADNI/AD_NC'

    #Set Hyperparameters
    image_size = 256
    batch_size = 32

    # First Configure the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not found, using CPU")
    print(torch.cuda.get_device_name(0))

    print("Start DataLoading ...")

    # the mean and std values are hardcoded here, previously calculated in utils.py from the training data
    transform = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1156, 0.1156, 0.1156), (0.2253, 0.2253, 0.2253))
        ]),
        'test': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.1156, 0.1156, 0.1156), (0.2253, 0.2253, 0.2253))
        ]),
    }

     # Create the complete dataset for training
    complete_train_dataset = ADNIDataset(data_dir=os.path.join(data_directory, 'train'), transform=transform, mode='train')

    # Split the training dataset: 80% training, 20% validation
    train_size = int(0.8 * len(complete_train_dataset))
    val_size = len(complete_train_dataset) - train_size
    train_dataset, val_dataset = random_split(complete_train_dataset, [train_size, val_size])

    # Create train, test, and validation datasets
    train_dataset = ADNIDataset(data_dir=os.path.join(data_directory, 'train'), transform=transform, mode='train')
    test_dataset = ADNIDataset(data_dir=os.path.join(data_directory, 'test'), transform=transform, mode='test')


    # DataLoader for batching
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    

    # Print dataset statistics
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of testing images: {len(test_dataset)}")
    print(f"Classes in dataset: AD (0), NC (1)")

     # iterate over DataLoader
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    print(f"Batch size: {images.size(0)}")
    print(f"Image size: {images.size()}")  # (batch_size, channels, height, width)
    print(f"Labels: {labels}")  # Tensor containing the class indices

    # visual to check if we can see the images
    def imshow(img):
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()

    imshow(torchvision.utils.make_grid(images[:4]))


