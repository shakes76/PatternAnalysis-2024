import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MedicalImageDataset  # Make sure this import matches your dataset module
from modules import UNet  # Import your U-Net model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Paths to your dataset
train_image_dir = 'C:/Users/rober/Desktop/COMP3710/keras_slices_seg_train'  # Adjust this path
train_label_dir = 'C:/Users/rober/Desktop/COMP3710/keras_slices_train'  # Adjust this path

# Load dataset
train_dataset = MedicalImageDataset(train_image_dir, train_label_dir, device)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Initialize model, optimizer, and loss function
in_channels = 1  # Adjust based on your input image channels
out_channels = 1  # Adjust based on the number of output channels
model = UNet(in_channels=in_channels, out_channels=out_channels).to(device)  # Move model to device

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()  # Assuming binary segmentation, adjust as needed

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    print(f'Starting epoch {epoch + 1}/{num_epochs}')

    for images, labels in train_loader:
        try:
            # Move images and labels to device
            images, labels = images.to(device), labels.to(device)

            # Debugging: Print shapes and devices
            print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"Images device: {images.device}, Labels device: {labels.device}")

            # Forward pass
            outputs = model(images)  # Ensure both are on the same device

            # Debugging: Print outputs shape
            print(f"Outputs shape: {outputs.shape}")

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        except Exception as e:
            print(f"Error during training: {e}")
