# Dataset pre-processing and storage

import yaml
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
import logging
from PIL import Image   
from sklearn.model_selection import train_test_split
import random

# Class for storing and pre-processing the ISIC Dataset
class ISICDataset(Dataset):
    def __init__(self, dir, ratio, transform=None, mode="train", first_run=True):
        
        self._dir = dir
        self._imgs = []
        self._mode = mode
        self._ratio = ratio
        self._transform = transform
        
        # Preprocess images and labels if first time running model
        if first_run:
            self.preprocess()
        
        # Get benign and malignant images and store into labelled reference arrays
        benign_refs = self.generate_img_references(os.path.join(self._dir, 'benign'), 0)
        malignant_refs = self.generate_img_references(os.path.join(self._dir, 'malignant'), 1)
        
        # Merge references into singular array
        img_refs = benign_refs + malignant_refs
        
        # Split data into training and validation sets based on the ratio
        # Data stratified to account for unbalanced dataset label ratio
        img_train, img_val = train_test_split(img_refs, train_size=ratio, stratify=[label for _, label in img_refs])
        
        self._imgs = img_refs
        
        return
    
    def preprocess(self):
        return
    
    def __len__(self):
        return len(self._imgs)
    
    # Retrieve specific data from dataset based on index
    def __getitem__(self, index):
        img_ref, label = self._imgs[index]
        
        # Generate random indexes for positive and negative images
        index_pos = random.choice()
        index_neg = random.choice()
        
        # Acquire images
        img_anchor = self.load_img(img_ref)
        img_pos = self.load_img(self._imgs[index_pos][0])
        img_neg = self.load_img(self._imgs[index_neg][0])
        
        # Apply transformations if present
        if self._transform is not None:
            img_anchor = self._transform(img_anchor)
            img_pos = self._transform(img_pos)
            img_neg = self._transform(img_neg)
            
        return img_anchor, label, img_pos, img_neg
    
    # Generate an array that maps a image (through its path) to its label
    # Assuming that all images within the given directory belongs to label
    def generate_img_references(self, dir, label):
        refs = []
        for file in os.listdir(dir):
            if file.endswith(".png", ".jpg", ".svg", ".jpeg"): # Common image extensions DEBUG - MENTION IN README
                img_ref = os.path.join(dir, file)
                refs.append((img_ref, label))
        return refs
    
    # Load an image in RGB format through its reference path
    def load_img(self, ref):
        img = Image.open(ref)
        img = img.convert("RGB")
        return img
    
# Create dataloaders for testing and validating
def generate_dataloader(dir, batch_size=16, ratio=0.75, n_workers=4):
    return
    
# Load relavent dataloading parameters from config.yaml
def load_params(self):
    return
    
if __name__ == "__main__":
    pass