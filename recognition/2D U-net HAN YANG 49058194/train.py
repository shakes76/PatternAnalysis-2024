import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import UNet
from dataset import ProstateMRIDataset
import matplotlib.pyplot as plt

# Check if the device GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training model
def train_model(root_dir, num_epochs=30, lr=0.001):
    # Load dataset
    dataset = ProstateMRIDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Create models, define loss functions, and optimizers
    model = UNet(n_channels=1, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []

    # Training cycle
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for num, images in enumerate(dataloader):
            if num%10==0:
                print(f"The {num} batches is processing.")
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("end of one epoch.")

        # Record and output losses
        epoch_loss = running_loss / len(dataloader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'unet_model.pth')

    # Plotting training loss in PC
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    root_dir = '/home/Student/s4905819/HipMRI_study_keras_slices_data/processed_nii_files'  
    train_model(root_dir)
