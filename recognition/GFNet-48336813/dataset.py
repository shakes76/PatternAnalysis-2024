"""
This script handles the creation and management of datasets for training, validation, and testing the GFNet model,
specifically for binary classification tasks (AD vs NC). It defines a custom PyTorch dataset class for loading images
from the ADNI dataset and applies necessary transformations such as resizing, normalization, and data augmentation.

The script includes functions to build datasets for different phases (train, val, test) and compute statistics like 
mean and standard deviation, which are crucial for preprocessing.

@brief: Dataset management and preprocessing for the GFNet model.
@author: Sean Bourchier
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# Mean and standard deviation values for different splits of the ADNI dataset
ADNI_DEFAULT_MEAN_TRAIN = 0.19868804514408112
ADNI_DEFAULT_STD_TRAIN = 0.24770835041999817
ADNI_DEFAULT_MEAN_VAL = 0.12288379669189453
ADNI_DEFAULT_STD_VAL = 0.2244586944580078
ADNI_DEFAULT_MEAN_TEST = 0.12404339015483856
ADNI_DEFAULT_STD_TEST = 0.2250228226184845

class ADNI_Dataset(Dataset):
    """
    Custom dataset for loading ADNI data (NC and AD images).
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., 'data/').
            split (str): Dataset split to load ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on each image.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = ['NC', 'AD']  # Class labels: NC (Normal Cognitive), AD (Alzheimer's Disease)
        self.images = []
        self.labels = []

        # Populate lists of image paths and corresponding labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)  # 0 for NC, 1 for AD

        print(f'ADNI Dataset with {len(self.images)} instances for {split} split')

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image, label) where image is a transformed image tensor and label is an integer.
        """
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load the image and convert it to grayscale
        image = Image.open(img_path).convert('L')

        # Apply the transform if provided
        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(split, args):
    """
    Builds the dataset and returns it along with the number of classes.

    Args:
        split (str): The data split to load ('train', 'val', 'test').
        args: Parsed command-line arguments.

    Returns:
        tuple: (dataset, nb_classes) where nb_classes is the number of output classes.
    """
    transform = build_transform(split, args)
    if args.data_set == 'ADNI':
        dataset = ADNI_Dataset(root_dir=args.data_path, split=split, transform=transform)
        nb_classes = 2
    else:
        raise NotImplementedError(f"Dataset '{args.data_set}' is not implemented.")
    
    return dataset, nb_classes


def build_transform(split, args):
    """
    Builds the appropriate image transformation pipeline for the given dataset split.

    Args:
        split (str): The data split ('train', 'val', or 'test').
        args: Parsed command-line arguments containing the input size.

    Returns:
        transforms.Compose: The composed transformation pipeline.
    """
    if split == "train":
        # Transformation for training data
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=ADNI_DEFAULT_MEAN_TRAIN, std=ADNI_DEFAULT_STD_TRAIN),
        ])
    elif split == "val":
        # Transformation for validation data
        transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=ADNI_DEFAULT_MEAN_VAL, std=ADNI_DEFAULT_STD_VAL),
        ])
    elif split == "test":
        # Transformation for test data
        transform = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.CenterCrop(args.input_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=ADNI_DEFAULT_MEAN_TEST, std=ADNI_DEFAULT_STD_TEST),
        ])
    else:
        raise NotImplementedError(f"Split '{split}' is not implemented.")
    
    return transform


def get_mean_and_std(dataset):
    """
    Calculates the mean and standard deviation of a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset to compute statistics from.

    Returns:
        tuple: (mean, std) of the dataset.
    """
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in loader:
        # Flatten the images into a single dimension per batch
        images = images.view(images.size(0), -1)  # (batch_size, width*height)
        batch_samples = images.size(0)  # Number of images in the current batch
        total_images_count += batch_samples

        # Compute the sum and squared sum for each batch
        mean += images.mean(1).sum()
        std += images.std(1).sum()
    
    # Calculate global mean and standard deviation
    mean /= total_images_count
    std /= total_images_count
    
    return mean.item(), std.item()

