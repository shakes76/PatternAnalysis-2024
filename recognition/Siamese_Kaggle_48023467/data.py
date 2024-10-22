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
import shutil

# Class for storing and pre-processing the ISIC Dataset
class ISICDataset(Dataset):
    def __init__(self, img_dir, output_dir, label_csv, split_ratio, oversample_ratio, transform=None, mode="train", first_run=True):
        
        self._dir = output_dir
        self._imgs = []
        self._mode = mode
        self._split_ratio = split_ratio
        self._oversample_ratio = oversample_ratio
        self._transform = transform
        
        # Preprocess images and labels if first time running model
        if first_run:
            self.preprocess(label_csv, img_dir, output_dir)
        
        # Get benign and malignant images and store into labelled reference arrays
        benign_refs = self.generate_img_references(os.path.join(self._dir, 'benign'), 0)
        malignant_refs = self.generate_img_references(os.path.join(self._dir, 'malignant'), 1)
        
        # Merge references into singular array
        img_refs = benign_refs + malignant_refs
        
        # Split data into training and validation sets based on the ratio
        # Data stratified to account for unbalanced dataset label ratio
        # DEBUG - MENTION RANDOM STATE FOR REPRODUCEABILITY
        img_train, img_val = train_test_split(img_refs, train_size=split_ratio, stratify=[label for _, label in img_refs], random_state=27)
        
        # Balance data in training mode by oversampling
        if mode == 'train':
            # Manual oversampling due to data storage method
            benign_train, malignant_train = [], []
            self._imgs = benign_train
            for img in img_train:
                benign_train.append(img) if img[1] == 0 else malignant_train.append(img)
                
            benign_count = len(benign_train)
            malignant_count = len(malignant_train)
            
            # Calculate amount minority class (malignant) needs to increase by
            oversample_count = int((benign_count * oversample_ratio) - malignant_count)
            
            # Randomly resample if oversample count is valid (>0)
            if oversample_count > 0:
                for _ in range(0, oversample_count):
                    sample = random.choice(malignant_train)
                    self._imgs.append(sample)
                    
        else:
            self._imgs = img_val
        
        random.shuffle(self._imgs)
        
        # Split image array into seperate arrays based on labels
        benign_indices, malignant_indices = [], []
        i = 0
        for img in self._imgs:
            benign_indices.append(i) if img[1] == 0 else malignant_indices.append(i)
            i += 1
            
        self._imgs_split = {0:benign_indices, 1:malignant_indices}
        return
    
    # Seperate and group image data based on labels
    def preprocess(self, labelCSV, img_path, output_dir):
        print("Initial run: Preprocessing data...")
        df = pd.read_csv(labelCSV)
        
        output_benign = os.path.join(output_dir, "benign")
        output_malignant = os.path.join(output_dir, "malignant")
        
        # Generate output directories if necessary
        if not os.path.exists(output_benign):
            os.makedirs(output_benign)
        if not os.path.exists(output_malignant):
            os.makedirs(output_malignant)
            
        for _, row in df.iterrows():
            img_name = row["isic_id"] + ".jpg"
            img_src = os.path.join(img_path, img_name)
            
            if os.path.isfile(img_src):
            
                # Seperate based on label
                if row["target"] == 0:
                    img_dst = os.path.join(output_benign, img_name)
                elif row["target"] == 1:
                    img_dst = os.path.join(output_malignant, img_name)
                
                shutil.copy(img_src, img_dst)
            
        return
    
    def __len__(self):
        return len(self._imgs)
    
    # Retrieve specific data from dataset based on index
    def __getitem__(self, index):
        img_ref, label = self._imgs[index]
        
        # Generate random indexes for positive and negative images
        index_pos = random.choice(self._imgs_split[label])
        while index_pos == index:
            index_pos = random.choice(self._imgs_split[label])
            
        index_neg = random.choice(self._imgs_split[1-label])
        
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
            if file.endswith((".png", ".jpg", ".svg")): # Common image extensions DEBUG - MENTION IN README
                img_ref = os.path.join(dir, file)
                refs.append((img_ref, label))
        return refs
    
    # Load an image in RGB format through its reference path
    def load_img(self, ref):
        img = Image.open(ref)
        img = img.convert("RGB")
        return img
    
# Create dataloaders for testing and validating
def generate_dataloaders(dir, batch_size=16, ratio=0.75, n_workers=4):
    # Augmentation transformations applied only to training dataset
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load params
    _, label_csv, output_dir, _, oversample_ratio, initial_run, _ = load_params()
    
    # Generate datasets
    train_dataset = ISICDataset(img_dir=dir, output_dir=output_dir, label_csv=label_csv, split_ratio=ratio, oversample_ratio=oversample_ratio, transform=train_transform, mode="train", first_run=initial_run)
    val_dataset = ISICDataset(img_dir=dir, output_dir=output_dir, label_csv=label_csv, split_ratio=ratio, oversample_ratio=oversample_ratio, transform=val_transform, mode="test", first_run=initial_run)
    
    # Generate dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=n_workers,
        batch_size=batch_size,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        num_workers=n_workers,
        batch_size=batch_size,
        pin_memory=True
    )

    return train_loader, val_loader
    
# Load relavent dataloading parameters from config.yaml
def load_params():
    with open("config.yaml", 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    data_dir = data["DatasetImageDir"]
    label_csv = data["LabelCSV"]
    output_dir = data["OutputDir"]
    train_ratio = data["TrainTestRario"]
    oversample_ratio = data["OversampleRatio"]
    initial_run = data["FirstRun"]
    batch_size = data["BatchSize"]
    
    return data_dir, label_csv, output_dir, train_ratio, oversample_ratio, initial_run, batch_size
    
if __name__ == "__main__":
    # For standalone data processing and storage testing, as well as manual resource validation
    
    # Load params
    data_dir, _, output_dir, train_ratio, _, _, batch_size = load_params()
    
    # Get data
    print("Loading data...")
    train_loader, val_loader = generate_dataloaders(data_dir, ratio=train_ratio, batch_size=batch_size)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Verify data presence and count
    benign_count_train = len([label for label in train_dataset._imgs if label[1] == 0])
    malignant_count_train = len([label for label in train_dataset._imgs if label[1] == 1])
    benign_count_val = len([label for label in val_dataset._imgs if label[1] == 0])
    malignant_count_val = len([label for label in val_dataset._imgs if label[1] == 1])

    print("\nData loading successful:")
    print(f"Training data - Total: {len(train_dataset)}, Benign: {benign_count_train}, Malignant: {malignant_count_train}")
    print(f"Validation data - Total: {len(val_dataset)}, Benign: {benign_count_val}, Malignant: {malignant_count_val}")