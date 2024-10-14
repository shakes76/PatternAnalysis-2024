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
    







