# dataset.py

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from sklearn.utils import class_weight
from collections import defaultdict

def get_data_loaders(data_dir, meta_data_path, batch_size=32, img_size=224, val_split=0.2, num_workers=4):
    """
    Creates training, validation, and test data loaders with a patient-level split.

    Args:
        data_dir (str): Path to the ADNI dataset directory.
        meta_data_path (str): Path to the meta_data_with_label.json file.
        batch_size (int): Batch size for data loaders.
        img_size (int): Size to resize the images.
        val_split (float): Fraction of training data to use for validation.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
        list: List of class names.
        torch.Tensor: Class weights for handling imbalance.
    """

    # Define image transformations
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                             std=[0.229, 0.224, 0.225])   # ImageNet stds
    ])

    test_val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load metadata to perform patient-level split
    with open(meta_data_path, 'r') as f:
        meta_data = json.load(f)

    # Group images by patient
    patient_to_images = defaultdict(list)
    for item in meta_data:
        patient_id = item['patient_id']
        image_path = os.path.join(data_dir, item['image_path'])
        patient_to_images[patient_id].append(image_path)

    # Split patients into train, validation, and test sets
    all_patients = list(patient_to_images.keys())
    np.random.shuffle(all_patients)

    # Define splits
    num_patients = len(all_patients)
    num_test_patients = int(0.2 * num_patients)  # 20% for testing
    num_val_patients = int(val_split * (num_patients - num_test_patients))
    num_train_patients = num_patients - num_test_patients - num_val_patients

    # Partition patients
    train_patients = all_patients[:num_train_patients]
    val_patients = all_patients[num_train_patients:num_train_patients + num_val_patients]
    test_patients = all_patients[num_train_patients + num_val_patients:]

    # Collect image paths for each split
    train_images = [img for p in train_patients for img in patient_to_images[p]]
    val_images = [img for p in val_patients for img in patient_to_images[p]]
    test_images = [img for p in test_patients for img in patient_to_images[p]]

    # Create custom datasets from image paths
    train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=data_dir, transform=test_val_transforms)
    test_dataset = datasets.ImageFolder(root=data_dir, transform=test_val_transforms)

    # Helper function to filter dataset based on image paths
    def filter_dataset(dataset, image_paths):
        indices = [i for i, (path, _) in enumerate(dataset.samples) if path in image_paths]
        return Subset(dataset, indices)

    # Filter datasets using the collected image paths
    train_dataset = filter_dataset(train_dataset, train_images)
    val_dataset = filter_dataset(val_dataset, val_images)
    test_dataset = filter_dataset(test_dataset, test_images)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class names
    class_names = train_dataset.dataset.classes  # Access original dataset classes

    # Compute class weights for handling class imbalance
    labels = [label for _, label in train_dataset]
    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes=np.unique(labels),
                                                      y=labels)
    # Convert computed class weights to PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, class_names, class_weights

