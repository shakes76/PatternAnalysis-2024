import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def extract_zip(zip_path, extract_to):
    '''Extracts the zip file into the data folder.'''
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.") 

class ADNIDataset(Dataset):
    '''Custom dataset class inherited from pytorch dataset class.
        It is used to load and preprocess the dataset'''
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # label 1 for AD and 0 for normal
        self._load_data('AD', 1)
        self._load_data('NC', 0)
    
    def _load_data (self, folder_name, label):
        folder_path = os.path.join(self.data_dir, folder_name)
        for file in os.listdir(folder_path):
            if file .endswith('.png'):
                img_path = os.path.join(folder_path, file)
                self.data.append(img_path)
                self.labels.append(label)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)
        
        return image, label
    
def get_data_loaders(zip_path, extract_to, batch_size=32, train_split = 0.85):
    """ Loading the training, testing and validating dataset.
    The training and validating dataset are seperated from 
    the original train dataset"""
    
    extract_zip(zip_path, extract_to)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = ADNIDataset(os.path.join(extract_to, 'AD_NC/train'), transform= transform)
    test_dataset = ADNIDataset(os.path.join(extract_to, 'AD_NC/test'), transform= transform)

    train_size = int(train_split*len(train_dataset))
    val_size = len(train_dataset)-train_size

    train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    return train_loader, val_loader, test_loader
    







