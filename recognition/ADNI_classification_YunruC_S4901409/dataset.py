import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
print(transforms.ToTensor)

def extract_zip(zip_path, extract_to):
    '''Extracts the zip file into the data folder.'''
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.") 
        
        ensure_folders_exist(extract_to)

def ensure_folders_exist(base_folder):
    """Ensure that the required folders exist"""
    required_folders = ["train/AD", "train/NM",
        "test/AD", "test/NM"]
    
    for folder in required_folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.ecists(folder_path):
            os.makedirs(folder_path)
    





