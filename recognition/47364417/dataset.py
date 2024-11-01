import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from PIL import Image

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Returns training, validation, and testing dataloaders along with class names.
    """
    # Define transformations for training, validation, and testing
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }


class BrainDataset(Dataset):
    """
    A custom Dataset class for loading and preprocessing brain images for generation and classification.

    Attributes:
        image_paths (list): List of file paths for brain image types.
        transform (callable, optional): A transformation to apply to the images.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = 0 if 'NC' in image_path else 1
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
    
class DataHandler:
    """
    Class to handle training and testing data set paths.
    """
    @staticmethod
    def load_data_paths():
        train_ad = 'recognition/47364417/AD_NC/train/AD'
        train_nc = 'recognition/47364417/AD_NC/train/NC'
        test_ad = 'recognition/47364417/AD_NC/test/AD'
        test_nc = 'recognition/47364417/AD_NC/test/NC'
        
        train_paths = [os.path.join(train_ad, img) for img in os.listdir(train_ad)] + \
                      [os.path.join(train_nc, img) for img in os.listdir(train_nc)]
        test_paths = [os.path.join(test_ad, img) for img in os.listdir(test_ad)] + \
                     [os.path.join(test_nc, img) for img in os.listdir(test_nc)]
        
        return train_paths, test_paths
    
    @staticmethod
    def _get_image_paths(directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        return [os.path.join(directory, img) for img in os.listdir(directory)]

# Basic transform applied to images for model input.
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])