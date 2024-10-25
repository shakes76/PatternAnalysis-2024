import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet
from dataset import ProstateCancerDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Check if GPU is available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
lr = 0.001
batch_size = 8
epochs = 50

# Initialize model, loss, optimizer
model = UNet().to(device)  # Move the model to the appropriate device (GPU/CPU)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load dataset directories (replace with your actual paths)
image_dir = r'1HipMRI_study_keras_slices_data/keras_slices_train'
mask_dir = r'1HipMRI_study_keras_slices_data/keras_slices_seg_train'

train_dataset = ProstateCancerDataset(image_dir, mask_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Define your model, loss, and optimizer (as before)
model = UNet(in_channels=1, out_channels=1).to(device)  # Example: Grayscale in, single class out
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
def train():
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss / len(train_loader):.4f}")

if __name__ == "__main__":
    train()