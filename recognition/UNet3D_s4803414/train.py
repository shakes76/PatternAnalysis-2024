import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRIDataset
from modules import UNet3D
from torchvision import transforms

IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_png_slices_train'
MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_png_slices_seg_train'
MODEL_SAVE_PATH = '/home/Student/s4803414/miniconda3/model/model.pth'

BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4

# Create dataset and dataloader
dataset = MRIDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model, loss function, and optimizer
model = UNet3D(in_channels=1, out_channels=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, masks in dataloader:
        # Move to GPU if available
        images = images.to('cuda') if torch.cuda.is_available() else images
        masks = masks.to('cuda') if torch.cuda.is_available() else masks

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))  # Squeeze to match output shape

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        # Print loss for the epoch
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(dataloader):.4f}')
