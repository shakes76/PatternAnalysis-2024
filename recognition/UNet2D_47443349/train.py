"""
train.py

Author: Alex Pitman
Student ID: 47443349
COMP3710 - HipMRI UNet2D Segmentation Project
Semester 2, 2024

Contains model training.
"""

from modules import UNet2D
from dataset import ProstateDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
from utils import TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR
import matplotlib.pyplot as plt
from utils import SEED, set_seed
from utils import CombinedLoss

# Set seed for reproducibility
set_seed(SEED)

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5
LABEL_WEIGHTS = torch.tensor([1.0, 1.0, 1.5, 1.5, 2.5, 2.5]).to(DEVICE)

# Data loading
transform_train = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
])

train_set = ProstateDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transforms=transform_train, early_stop=False, normImage=True)
validation_set = ProstateDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transforms=None, early_stop=False, normImage=True)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)

# Initialisations
model = UNet2D(in_channels=1, out_channels=6, initial_features=64, n_layers=4).to(DEVICE)
criterion = CombinedLoss(label_weights=LABEL_WEIGHTS, dice_weight=0.8)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Lists to store epoch losses for plotting
train_losses = []
validation_losses = []

# Training function
def train_function(loader, model, optimizer, criterion, device):
    """
    Training loop for one epoch of model training.
    """
    model.train()
    running_loss = 0.0
    total_step = len(loader)
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss tracking
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print("Step [{}/{}], Training Loss: {:.5f}".format(batch_idx + 1, total_step, loss.item()))
    # Store loss (average over epoch)
    train_losses.append(running_loss / len(loader))

# Validation function
def validate_function(loader, model, criterion, device):
    """
    Validation loop for one epoch of model validation
    """
    model.eval()
    running_loss = 0.0
    total_step = len(loader)
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Loss tracking
            running_loss += loss.item()
            if (batch_idx + 1) % 10 == 0:
                print("Step [{}/{}], Validation Loss: {:.5f}".format(batch_idx + 1, total_step, loss.item()))
    # Store loss (average over epoch)
    validation_losses.append(running_loss / len(loader))

# Run model training with validation
for epoch in range(NUM_EPOCHS):
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
    train_function(train_loader, model, optimizer, criterion, DEVICE)
    validate_function(validation_loader, model, criterion, DEVICE)

# Save model
torch.save(model.state_dict(), "UNet2D_Model.pth")

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, NUM_EPOCHS + 1), validation_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig("Loss Curves.png", format='png')
plt.show()
