"""
REFERENCES:

(1) This code was developed with assistance from the Claude AI assistant,
    created by Anthropic, PBC. Claude provided guidance on implementing
    StyleGAN2 architecture and training procedures.

    Date of assistance: 8-21/10/2024
    Claude version: Claude-3.5 Sonnet
    For more information about Claude: https://www.anthropic.com
"""


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain image dataset.
    Handles grayscale JPEG images - size 256x240 (pad to 256x256)
    Labeled in AD (Alzheimer's Disease) and NC (Normal Control) categories.
    """
    def __init__(self, 
                 root_dir,          # Directory with images.
                 split='train',     # Use train or test images?
                 transform=None     # Optional transform to sample.
                ):   
        self.root_dir = root_dir
        self.split = split
        self.image_paths = []
        self.labels = []
        
        # Define default transformations if None
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((256, 240)), # making sure images fit expected size
                transforms.Pad((8, 0, 8, 0), fill=0),  # Pad to 256x256
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) # [-1, 1] normalisation - see if this makes difference
                # transforms.Normalize([0.1156], [0.2200])  # Mean, stddev from AD_NC train dataset
            ])
        else:
            self.transform = transform
        
        # Collect image paths and labels
        for category in ['AD', 'NC']:
            category_path = os.path.join(self.root_dir, self.split, category)
            for img in os.listdir(category_path):
                if img.endswith('.jpeg') or img.endswith('.jpg'):
                    self.image_paths.append(os.path.join(category_path, img))
                    self.labels.append(0 if category == 'AD' else 1)  # 0 for AD, 1 for NC
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label