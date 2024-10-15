'''
Containing the data loader for loading and preprocessing your data

Created by: Shogo Terashima
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class DataPreprocessing:
    '''
    This is to load and preprocess given dadta.
    Args:
        dataset_path (string)
        batch_size (int)
        num_workers (int)
    '''

    def __init__(self, dataset_path, batch_size=64, num_workers=4):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Add augumentation for training set
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1156], std=[0.2198])  # pre-calculated mean and std
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1156], std=[0.2198])  # pre-calculated mean and std
        ])

# def calculateNormaliesParameters(loader):
#     mean = std = num_images = 0
#     for images, labels in loader:
#         # images: (N = batch size , C = 1, H, W)
#         batch_num_images = images.size(0)
#         images = images.view(batch_num_images, images.size(1), -1)  # (N, C = 1, H * W)
#         mean += images.float().mean(2).sum(0)
#         std += images.float().std(2).sum(0)
#         num_images += batch_num_images

#     mean /= num_images
#     std /= num_images
#     return mean, std

    def get_train_val_loaders(self, val_split=0.2):
        """
        Here, I want to split train to train and validation sets.
        Args:
            val_split (float): Ratio of data to be used as validation
        Returns:
            train_loader, val_loader: DataLoaders for training and validation
        """
        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.test_transform)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_dataset.dataset.transform = self.train_transform # augument only for train
        val_dataset.dataset.transform = self.test_transform

        # Create DataLoader for both train and validation sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) # no need to shuffle for validation


        return train_loader, val_loader

    def get_test_loader(self):
        '''
        This is for lading test set.
        Return:
            test loader
        '''
        test_dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) # no need to shuffle for test
        return test_loader


# Example usage
train_dataset_path = "../dataset/AD_NC/train"
test_dataset_path = "../dataset/AD_NC/test"
train_processor = DataPreprocessing(train_dataset_path, batch_size=128)
test_processor = DataPreprocessing(test_dataset_path, batch_size=128)
train_loader, val_loader = train_processor.get_train_val_loaders(val_split=0.2)
test_loader = test_processor.get_test_loader()



