'''
dataset.py
@brief  Script load the datasets for GFNet
@author  Benjamin Jorgensen - s4717300
@date   18/10/2024
'''
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import Generator
from torch.utils.data import random_split
import os


class GFNetDataloader():
    """
    Custom dataloader for loading images into the GFNet. 
    Provides image augmentation as well as splitting data into training validation and test sets

    @params batch_size: Number of images in a batch
    """
    def __init__(self, batch_size=64):
        self.batch_size     = batch_size
        self._mean          = 0.0
        self._std           = 0.0
        self._total_images  = 0
        self.train_loader   = None
        self.val_loader     = None
        self.img_size       = None
        self.datapath       = None
        self.n_classes      = 0

        self.gen = Generator().manual_seed(4717300) # Student number as seed

    def load(self, datapath):
        """
        Loads training and validation data from a directory, and then transforms tem
        """
        self.datapath = datapath

        # Convert all images into greyscale
        init_trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # Normalise the data
        dataset = datasets.ImageFolder(root=os.path.join(datapath, 'train'), transform=init_trans)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._mean, self._std, self._total_images = get_distribution(loader)

        # Get the shape of the images
        images, _ = next(iter(loader))
        image_size = min(images.shape[-2:])

        # Transformation for the validation set: (No augmentation)
        val_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=self._mean, std=self._std)
                ])

        # Transformation for training set
        _transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),              # Flip horizontally
            transforms.RandomVerticalFlip(p=0.5),           # Flip vertically with 50% probability
            transforms.RandomRotation(30),                  # Random rotation within ±30 degrees
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness/contrast
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
            transforms.RandomCrop(image_size, padding=8, padding_mode='reflect'),
            transforms.Normalize(mean=self._mean, std=self._std),  # Use computed mean and stdk
            ])
        self.img_size = image_size

        train_data = datasets.ImageFolder(root=os.path.join(datapath, 'train'), transform=_transform)
        val_data = datasets.ImageFolder(root=os.path.join(datapath, 'test'), transform=val_trans)

        # Create a random subset of the validation set to be used as an estimate
        _, test_split = random_split(val_data, [0.9, 0.1], generator=self.gen)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_split, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.n_classes = len(train_data.classes)

    def load_validation(self, datapath):
        """
        Similar to load except we're only accessing a directory of images instead of the train/test splits
        """
        # Convert all images into greyscale
        init_trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        # Normalise the data
        dataset = datasets.ImageFolder(root=datapath, transform=init_trans)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self._mean, self._std, self._total_images = get_distribution(loader)

        # Get the shape of the images
        images, _ = next(iter(loader))
        image_size = min(images.shape[-2:])

        # Transformation for the validation set: (No augmentation)
        val_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.Normalize(mean=self._mean, std=self._std)
                ])
        self.img_size = image_size

        val_data = datasets.ImageFolder(root=os.path.join(datapath), transform=val_trans)


        self.val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        self.n_classes = len(val_data.classes)


    def transform_val(self, data, image_size):
        """
        Used for transforming single images using inference
        """
        val_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                ])
        return val_trans(data)

    def get_data(self):
        return self.train_loader, self.test_loader, self.val_loader

    def get_meta(self):
        return {"total_images": self._total_images,
                "mean": self._mean,
                "std": self._std,
                "img_size": self.img_size,
                "channels": 1,
                "n_classes": self.n_classes
                } 

def get_distribution(loader):
    """
    Calculates the mean pixel value and standard deviation

    @params loader: dataset to calculate the distribution for
    """
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        # Compute the mean and std per batch
        batch_samples = images.size(0)  # Get the number of images in the batch
        images = images.view(batch_samples, images.size(1), -1)  # Flatten the image
        mean += images.mean(2).sum(0)  # Sum mean across batch
        std += images.std(2).sum(0)    # Sum std across batch
        total_images += batch_samples

    # Final mean and std calculations
    mean /= total_images
    std /= total_images
    return mean, std, total_images
