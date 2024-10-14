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


        






