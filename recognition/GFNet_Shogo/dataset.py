'''
Containing the data loader for loading and preprocessing the dataset.
It also containing the method to calculate the mean and the standard deviation
that will be used for normalising the datasets.

Created by:     Shogo Terashima
ID:             S47779628
Last update:    24/10/2024
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

class CombinedPreprocessing:
    '''
    This class loads and preprocesses the combined train, validation, and test sets from given paths.
    Args:
        train_path (string): Path to the training dataset.
        test_path (string): Path to the testing dataset.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of worker threads for loading data.
        val_split (float): Ratio of data to be used as validation set.
        seed       (int): The seed for random split. Use for replicat results.
    '''

    def __init__(self, train_path, test_path, batch_size=64, num_workers=1, val_split=0.2, seed=20):
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        # Base transform for initial data loading
        self.transform_base = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def get_data_loaders(self):
        """
        This method combines the train and test datasets, shuffles, and splits them into train, validation, and test sets.
        Returns:
            train_loader, val_loader, test_loader
        """
        # Load train and test datasets
        train_dataset = datasets.ImageFolder(root=self.train_path, transform=self.transform_base)
        test_dataset = datasets.ImageFolder(root=self.test_path, transform=self.transform_base)

        # Concat and reproduce train and test sets
        combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        train_val_size = int(0.8 * len(combined_dataset))
        test_size = len(combined_dataset) - train_val_size
        train_val_dataset, test_dataset = random_split(combined_dataset, [train_val_size, test_size],
                                                       generator=torch.Generator().manual_seed(self.seed))

        # Split train_val into train and validation
        train_size = int((1 - self.val_split) * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(self.seed))

        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        mean, std = self.calculate_normalisation_parameters(loader)
        print(f'Calculated mean: {mean}, std: {std}')

        # Define train and validation transforms
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)), # adjust the image size to perfect square
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        self.validation_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        # Update transforms for the datasets
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.validation_transform
        test_dataset.dataset.transform = self.validation_transform

        # Create DataLoader for train, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader

    @staticmethod
    def calculate_normalisation_parameters(loader):
        '''
        The method to calculate the mean and the standard deviation.
        Args:
            loader: Datalodaer created with dataset.
        Returns:
            mean and standard deviation calculated from the dataset.
        '''
        mean = torch.zeros(1)
        std = torch.zeros(1)
        num_images = 0
        for images, labels in loader:
            batch_num_images = images.size(0)
            images = images.view(batch_num_images, images.size(1), -1)
            mean += images.float().mean(2).sum(0)
            std += images.float().std(2).sum(0)
            num_images += batch_num_images

        mean /= num_images
        std /= num_images
        return mean, std
