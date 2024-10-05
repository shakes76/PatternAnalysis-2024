from torch.utils.data import Dataset
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

class ADNIDataset(Dataset):
    def __init__(self, ad_path, nc_path):
        ad_files = os.listdir(ad_path)
        nc_files = os.listdir(nc_path)
        
        
        
        ad_files_labelled = [(os.path.join(ad_path, file), 1) for file in ad_files]
        nc_files_labelled = [(os.path.join(nc_path, file), 0) for file in nc_files]
        
        self.data = ad_files_labelled + nc_files_labelled
        
        self.augmentations = A.Compose([
            A.GaussNoise(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=40, p=0.5),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        file, label = self.data[item]
        image = np.array(Image.open(file))
        image = self.augmentations(image=image)['image']
        
        return image, label
        