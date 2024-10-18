import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MRIDataset
from modules import UNet3D
from torchvision import transforms
from torch.amp import autocast, GradScaler
import numpy as np

torch.cuda.empty_cache()

# Directories and parameters
IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs'
MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only'
MODEL_SAVE_PATH = '/home/Student/s4803414/miniconda3/model/model.pth'

BATCH_SIZE = 4  # Reduced batch size
EPOCHS = 5
LEARNING_RATE = 1e-4
SPLIT_RATIO = [0.8, 0.1, 0.1]  # Train, validation, and test split

# Create the dataset
dataset = MRIDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=None, augment=True)

# Calculate split sizes
train_size = int(SPLIT_RATIO[0] * len(dataset))
test_size = int(SPLIT_RATIO[1] * len(dataset))
val_size = len(dataset) - train_size - test_size

# Split dataset into training, testing, and validation sets
train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize the model, loss function, and optimizer
model = UNet3D(in_channels=1, out_channels=6)  # 6 classes in total
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the device


# Function to compute Dice score for each class
def dice_score(pred, target, num_classes):
    smooth = 1e-6  # Small constant to avoid division by zero
    pred = torch.argmax(pred, dim=1)  # Get predicted class for each pixel
    dice_per_class = []

    for i in range(num_classes):
        pred_i = (pred == i).float()  # Get the predicted mask for class i
        target_i = (target == i).float()  # Get the target mask for class i
        intersection = (pred_i * target_i).sum()  # Compute intersection
        dice = (2. * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)
        dice_per_class.append(dice.item())

    return dice_per_class


# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, masks in train_loader:
        # Move to GPU if available
        images = images.to(device)
        masks = masks.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))  # Squeeze to match output shape

        # Backward pass and optimization
        scaler.scale(loss).backward()  # Scale the loss
        scaler.step(optimizer)  # Update the parameters
        scaler.update()  # Update the scale for the next iteration

        # Accumulate loss
        running_loss += loss.item()

    # Print loss for the epoch
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}')

    # Validation loop with Dice score evaluation
    model.eval()
    val_loss = 0.0
    dice_scores = np.zeros(6)  # Assuming 6 classes
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))
            val_loss += loss.item()
            # Compute Dice scores for each class
            dice_per_class = dice_score(outputs, masks.squeeze(1), num_classes=6)
            dice_scores += np.array(dice_per_class)

    # Average the dice scores
    avg_dice_scores = dice_scores / len(val_loader)
    print(f'Validation Loss after Epoch [{epoch + 1}/{EPOCHS}]: {val_loss / len(val_loader):.4f}')
    print(f'Class-specific Dice Scores: {avg_dice_scores}')

# Testing loop
model.eval()  # Set model to evaluation mode for testing
test_loss = 0.0
dice_scores = np.zeros(6)  # Reset Dice scores

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))  # Squeeze to match output shape
        test_loss += loss.item()

        # Compute Dice scores for each class
        dice_per_class = dice_score(outputs, masks.squeeze(1), num_classes=6)
        dice_scores += np.array(dice_per_class)

# Average the dice scores
avg_test_dice_scores = dice_scores / len(test_loader)

# Print test loss and Dice scores
print(f'Test Loss: {test_loss / len(test_loader):.4f}')
print(f'Test Class-specific Dice Scores: {avg_test_dice_scores}')

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)  # Create model directory if it doesn't exist
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f'Model saved to {MODEL_SAVE_PATH}')
