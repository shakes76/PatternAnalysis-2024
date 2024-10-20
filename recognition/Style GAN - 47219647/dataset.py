import numpy as np
import torch
import tensorflow as tf
from torchvision import  datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


def one_hot_encode(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes)

def data_set_creator():
    augmentation_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        # No vertical flipping was applied
        transforms.RandomHorizontalFlip(),
        # To account for skewing
        transforms.RandomRotation(30),
        # Coloring was altered to stop overfitting
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    # Change the directory to what is needed
    data_dir = 'recognition/Style GAN - 47219647/AD_NC/test'
    
    dataset = datasets.ImageFolder(root=data_dir, transform=augmentation_transforms)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)

    num_classes = len(dataset.classes)  

    def get_one_hot_encoded_loader():
        for batch_images, batch_labels in data_loader:
            one_hot_labels = one_hot_encode(batch_labels, num_classes)
            yield batch_images, one_hot_labels

    return get_one_hot_encoded_loader()



