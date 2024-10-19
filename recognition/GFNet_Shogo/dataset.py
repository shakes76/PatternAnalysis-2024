'''
Containing the data loader for loading and preprocessing your data

Created by: Shogo Terashima
'''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split


class TrainPreprocessing:
    '''
    This class is to loading train and validation sets from a dataset.
    Args:
        dataset_path (string)
        batch_size (int)
        num_workers (int)
    '''
    def __init__(self, dataset_path, batch_size=64, num_workers=1):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform_base = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])


    def get_train_val_loaders(self, val_split=0.2):
        """
        This method is to splits the train set into train and validation sets.
        Args:
            val_split (float): Ratio of data to be used as validation
        Returns:
            train_loader, val_loader
        """
        # calculate normalisation parameters
        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transform_base)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        mean, std = self.calculateNormalisationParameters(loader)
        print(f'Calculated mean: {mean}, std: {std}')

        # Define Train transoform
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1)
            ),
            transforms.ColorJitter(contrast=0.8, brightness=0.1), 
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])


        # Validation transform
        self.validation_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        # Split train to train and validation
        # val_size = int(len(dataset) * val_split)
        # train_size = len(dataset) - val_size
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        indices = list(range(len(dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=val_split, stratify=[label for _, label in dataset.samples], random_state=20)

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_dataset.dataset.transform = self.train_transform # Augument data for train
        val_dataset.dataset.transform = self.validation_transform  # No augmentation for validation

        # Create DataLoader for both train and validation sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)  # No need to shuffle for validation

        return train_loader, val_loader
    
    @staticmethod
    def calculateNormalisationParameters(loader):
        mean = torch.zeros(1)
        std = torch.zeros(1)
        num_images = 0 
        for images, labels in loader:
            # images: (N = batch size , C = 1, H, W)
            batch_num_images = images.size(0)
            images = images.view(batch_num_images, images.size(1), -1)  # (B, C, H * W)
            mean += images.float().mean(2).sum(0)
            std += images.float().std(2).sum(0)
            num_images += batch_num_images

        mean /= num_images
        std /= num_images
        return mean, std

class TestPreprocessing:
    '''
    This class is to load and preprocess test set
    Args:
        dataset_path (string)
        batch_size (int)
        num_workers (int)
    '''

    def __init__(self, dataset_path, batch_size=64, num_workers=1):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # No augumentation needed for test set
        self.transform_base = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def get_test_loader(self):
        '''
        Loads the test dataset.
        Returns:
            test_loader: DataLoader for the test set
        '''
        dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.transform_base)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        mean, std = self.calculateNormalisationParameters(loader)
        print(f'Calculated mean: {mean}, std: {std}')

        self.test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 8, 0, 8)),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist())
        ])

        test_dataset = datasets.ImageFolder(root=self.dataset_path, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)  # No shuffle for test
        
        return test_loader

    @staticmethod
    def calculateNormalisationParameters(loader):
        mean = torch.zeros(1)
        std = torch.zeros(1)
        num_images = 0
        for images, labels in loader:
            # images: (N = batch size , C = 1, H, W)
            batch_num_images = images.size(0)
            images = images.view(batch_num_images, images.size(1), -1)  # (B, C, H * W)
            mean += images.float().mean(2).sum(0)
            std += images.float().std(2).sum(0)
            num_images += batch_num_images

        mean /= num_images
        std /= num_images
        return mean, std


