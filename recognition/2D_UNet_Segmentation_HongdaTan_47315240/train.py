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
batch_size = 4
epochs = 50

# Initialize model, loss, optimizer
model = UNet().to(device)  # Move the model to the appropriate device (GPU/CPU)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load dataset directories (replace with your actual paths)
image_dir = r'C:/Users/11vac/Desktop/3710 Report/HipMRI_study_keras_slices_data/keras_slices_train'
mask_dir = r'C:/Users/11vac/Desktop/3710 Report/HipMRI_study_keras_slices_data/keras_slices_seg_train'

# Split data into training and validation sets
image_paths = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
mask_paths = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

train_images, val_images, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.15, random_state=42)

# Create dataset instances
train_dataset = ProstateCancerDataset(image_dir, mask_dir)
val_dataset = ProstateCancerDataset(image_dir, mask_dir)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Training loop
def train():
    train_loss = []
    val_loss = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)  # Move data to the appropriate device
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_loss.append(avg_train_loss)
        
        # Validate
        model.eval()
        running_val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)  # Move data to the appropriate device
                outputs = model(images)
                loss = criterion(outputs, masks)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save the model checkpoint
    torch.save(model.state_dict(), 'unet_model.pth')

    # Plot losses
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()
