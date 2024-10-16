import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRIDataset
from modules import UNet3D
from torchvision import transforms
from torch.amp import autocast, GradScaler

IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs'
MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only'
MODEL_SAVE_PATH = '/home/Student/s4803414/miniconda3/model/model.pth'

BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-4

# Create dataset and dataloader
dataset = MRIDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=None)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet3D(in_channels=1, out_channels=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the device once

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, masks in dataloader:
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
    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(dataloader):.4f}')

# Save the model
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)  # Create model directory if it doesn't exist
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f'Model saved to {MODEL_SAVE_PATH}')
