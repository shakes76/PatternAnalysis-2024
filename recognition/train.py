# TRAIN.PY

# IMPORTS
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import SimpleUNet
from dataset import SegmentationData  # Ensure this is the path to your SegmentationData class
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

# Datasets directories
train_image_dir = './data/HipMRI_study_keras_slices_data/keras_slices_train'
train_label_dir = './data/HipMRI_study_keras_slices_data/keras_slices_seg_train'
val_image_dir = './data/HipMRI_study_keras_slices_data/keras_slices_validate'
val_label_dir = './data/HipMRI_study_keras_slices_data/keras_slices_seg_validate'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Datasets
print("Loading Training Data")
train_dataset = SegmentationData(
    train_image_dir, train_label_dir,
    norm_image=False, categorical=True, dtype=np.float32, augment=False
)
print("Loading Validation Data")
val_dataset = SegmentationData(
    val_image_dir, val_label_dir,
    norm_image=False, categorical=True, dtype=np.float32, augment=False
)

# Create dataloaders
print("Creating Training Dataloader")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
print("Creating Training Validation Dataloader")
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the model
n_channels = 1  # Assuming input images are grayscale
n_classes = train_dataset.num_classes  # Number of classes in the dataset
model = SimpleUNet(n_channels=n_channels, n_classes=n_classes, dropout_p=0.3)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)  # Shape: (batch_size, n_channels, H, W)
        labels = labels.to(device)  # Shape: (batch_size, n_classes, H, W)

        # For CrossEntropyLoss, labels need to be (batch_size, H, W) with class indices
        labels = labels.argmax(dim=1)  # Convert one-hot encoded labels to class indices

        # Forward pass
        outputs = model(images)  # Shape: (batch_size, n_classes, H, W)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    print(f"Training Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.argmax(dim=1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}")

    # Optionally, save the model checkpoint
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
