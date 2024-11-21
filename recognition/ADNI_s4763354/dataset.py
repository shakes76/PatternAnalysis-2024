'''
Author: Lok Yee Joey Cheung
This file contains the function for data loading, data augmentation, patient ID split in train and validation sets and optional class balancing.
'''

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def get_data_loaders(train_dir, test_dir, batch_size=32, val_ratio=0.2, random_seed=42):
    # Define data augmentation and transformation for training set
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Define data transformation for validation and testing
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the dataset
    full_dataset = datasets.ImageFolder(root=train_dir, transform=None)
    
    # Map each image to its corresponding patient ID 
    patient_to_indices = {}
    for idx, (path, _) in enumerate(full_dataset.samples):
        filename = os.path.basename(path)
        patient_id = filename.split('_')[0]
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)
    
    # List of all unique patient IDs
    all_patient_ids = list(patient_to_indices.keys())
    
    # Split patient IDs into train and validation sets
    train_patient_ids, val_patient_ids = train_test_split(
        all_patient_ids,
        test_size=val_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=[full_dataset.targets[patient_to_indices[pid][0]] for pid in all_patient_ids]
    )
    
    # Gather all image indices for training and validation
    train_indices = []
    for pid in train_patient_ids:
        train_indices.extend(patient_to_indices[pid]) 
    val_indices = []
    for pid in val_patient_ids:
        val_indices.extend(patient_to_indices[pid])

    # Create subsets for training and validation
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    # Assign transformations to the subsets
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_test_transform
    
    # (Optional) Calculate class weights for balanced sampling for class balancing
    train_labels = [full_dataset.targets[i] for i in train_indices]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1. / class_counts.float()
    sample_weights = [class_weights[t] for t in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Create data loader
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader
