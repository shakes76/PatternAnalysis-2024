import os
import torch
import pydicom
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ISICDICOMDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.dicom_files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    
    def __len__(self):
        return len(self.dicom_files)
    
    def __getitem__(self, idx):
        dicom_path = os.path.join(self.directory, self.dicom_files[idx])
        dicom_data = pydicom.dcmread(dicom_path)
        
        image = Image.fromarray(dicom_data.pixel_array)
        
        if self.transform:
            image = self.transform(image)
        
        label = 0
        return image, label

def create_dicom_dataloader(directory, batch_size=32, img_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Assuming grayscale images; adjust if RGB
    ])
    
    dataset = ISICDICOMDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

if __name__ == '__main__':
    data_dir = 'path_to_your_dicom_files'
    dataloader = create_dicom_dataloader(data_dir, batch_size=32)
    
    for images, labels in dataloader:
        print(images.shape, labels.shape)
