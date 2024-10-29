import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from modules import UNet  # Import the UNet from modules.py
from dataset import ProstateMRIDataset  # Ensure this points to your dataset class

# Define constants
INPUT_IMAGE_HEIGHT = 256  # Set this to your desired input image height
INPUT_IMAGE_WIDTH = 128   # Set this to your desired input image width

# Directories for training and validation data
IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'  # Path to your images
SEGMENTATION_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train'  # Path to your segmentation masks
VAL_IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'
VAL_SEGMENTATION_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate'

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_WORKERS = 1
TARGET_SHAPE = (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)

# Load data
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
seg_image_paths = [os.path.join(SEGMENTATION_DIR, f) for f in os.listdir(SEGMENTATION_DIR)]
train_img_paths, val_img_paths, train_seg_paths, val_seg_paths = train_test_split(
    image_paths, seg_image_paths, test_size=0.2, random_state=42
)

# Create datasets and dataloaders
train_dataset = ProstateMRIDataset(train_seg_paths, train_img_paths, normImage=True, target_shape=TARGET_SHAPE)
val_dataset = ProstateMRIDataset(val_seg_paths, val_img_paths, normImage=True, target_shape=TARGET_SHAPE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modify `encChannels` to start with 1 channel if input images have 1 channel (grayscale images)
model = UNet(encChannels=(1, 32, 64, 128), decChannels=(128, 64, 32), nbClasses=1, retainDim=True,
              outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)).to(device)

criterion = nn.BCEWithLogitsLoss()  # For binary segmentation
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, segs in tqdm(train_loader):
        images, segs = images.to(device), segs.to(device)

        # Check the input image shape to ensure it matches what the model expects
        print(f"Batch {images.shape}, {segs.shape}")  # Should show [BATCH_SIZE, 1, 256, 128]

        optimizer.zero_grad()
        outputs = model(images)

        # Adjusting the segmentation mask shape for the loss calculation
        loss = criterion(outputs, segs)  # Both outputs and segs should have the shape [BATCH_SIZE, 1, 256, 128]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}')

