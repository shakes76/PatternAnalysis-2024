import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from modules import UNet  # Import UNet class from modules.py
from dataset import ProstateMRIDataset  # Import ProstateMRIDataset class from dataset.py

# Model parameters
INPUT_IMAGE_HEIGHT = 256
INPUT_IMAGE_WIDTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 10
NUM_WORKERS = 4
TARGET_SHAPE = (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)

# Directories for training and validation data
IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train'
SEGMENTATION_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train'
VAL_IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate'
VAL_SEGMENTATION_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate'

# Load data paths
print("Loading image paths...")
image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)]
seg_image_paths = [os.path.join(SEGMENTATION_DIR, f) for f in os.listdir(SEGMENTATION_DIR)]
print(f"Total images: {len(image_paths)}, Total segmentations: {len(seg_image_paths)}")

train_img_paths, val_img_paths, train_seg_paths, val_seg_paths = train_test_split(
    image_paths, seg_image_paths, test_size=0.2, random_state=42
)
print(f"Training images: {len(train_img_paths)}, Validation images: {len(val_img_paths)}")

# Define transformations for data augmentation on training data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(15),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# Create datasets with augmentation for training
train_dataset = ProstateMRIDataset(train_seg_paths, train_img_paths, normImage=True, target_shape=TARGET_SHAPE, transform=train_transform)
val_dataset = ProstateMRIDataset(val_seg_paths, val_img_paths, normImage=True, target_shape=TARGET_SHAPE)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("DataLoader created successfully.")

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = UNet(in_channels=1, out_channels=1, retainDim=True, outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)).to(device)
print("Model initialized.")

# Define loss function, optimizer, and scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, min_lr=1e-6)

# Dice Score calculation
def dice_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()  # Threshold at 0.5 for binary output
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

# Tracking variables
best_dice = 0
train_losses, val_losses, dice_scores = [], [], []

for epoch in range(EPOCHS):
    print(f"\nStarting epoch {epoch+1}/{EPOCHS}...")
    model.train()
    running_loss = 0.0
    train_dice_scores = []

    for i, (images, segs) in enumerate(tqdm(train_loader)):
        images, segs = images.to(device), segs.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, segs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dice = dice_score(outputs, segs)
        train_dice_scores.append(dice.item())

    avg_train_loss = running_loss / len(train_loader)
    avg_train_dice = np.mean(train_dice_scores)
    train_losses.append(avg_train_loss)
    dice_scores.append(avg_train_dice)
    print(f'Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} - Dice Score: {avg_train_dice:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_dice_scores = []
    with torch.no_grad():
        for images, segs in val_loader:
            images, segs = images.to(device), segs.to(device)
            outputs = model(images)
            loss = criterion(outputs, segs)
            val_loss += loss.item()
            val_dice = dice_score(outputs, segs)
            val_dice_scores.append(val_dice.item())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = np.mean(val_dice_scores)
    val_losses.append(avg_val_loss)
    print(f'Epoch [{epoch+1}/{EPOCHS}] - Val Loss: {avg_val_loss:.4f} - Val Dice Score: {avg_val_dice:.4f}')

    # Adjust learning rate
    scheduler.step(avg_val_dice)

    # Check for best model
    if avg_val_dice > best_dice:
        print(f"Validation Dice Score improved ({best_dice:.4f} --> {avg_val_dice:.4f}). Saving model...")
        torch.save(model.state_dict(), 'best_model.pth')
        best_dice = avg_val_dice
        best_epoch = epoch

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()

# Plot Dice scores
plt.figure(figsize=(10, 5))
plt.plot(dice_scores, label='Dice Score')
plt.title('Dice Score Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()
plt.savefig('dice_score_plot.png')
plt.show()

