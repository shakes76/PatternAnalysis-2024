from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class BrainDataset(Dataset):
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
    
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])