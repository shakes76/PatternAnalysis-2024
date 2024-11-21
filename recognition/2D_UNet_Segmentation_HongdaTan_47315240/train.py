import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet
from dataset import ProstateCancerDataset
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

# Function to plot training loss
def plot_training_loss(train_loss):
    plt.plot(train_loss, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Time")
    plt.legend()
    plt.show()

# Training loop
def train():
    train_loss_per_epoch = []  # To store average loss per epoch

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

        # Calculate and store average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        train_loss_per_epoch.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}")

    # Plot training loss after completing all epochs
    plot_training_loss(train_loss_per_epoch)

    # Save the trained model
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved as 'unet_model.pth'.")

if __name__ == "__main__":
    train()
