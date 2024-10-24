import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from modules import UNet  
from dataset import create_dataloader  

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_epochs = 25
batch_size = 4
learning_rate = 1e-3
num_classes = 6  # Adjust this based on your specific number of classes

# Paths
image_directory = "C:/Users/sophi/Downloads/HipMRI_study_keras_slices_data/keras_slices_train"   #### WILL NEED TO CHANGE TO WORK FOR YOU
label_directory = "C:/Users/sophi/Downloads/HipMRI_study_keras_slices_data/keras_slices_seg_train" #### WILL NEED TO CHANGE TO WORK FOR YOU

# Create data loaders
train_loader = create_dataloader(image_directory, label_directory, batch_size=batch_size)

# Initialize model
model = UNet(in_channels=1, out_channels=num_classes).to(device)  # Adjust in_channels if needed

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use an appropriate loss for segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model():
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move to GPU if available
            optimizer.zero_grad()  # Zero the parameter gradients
            
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

if __name__ == "__main__":
    train_model()
