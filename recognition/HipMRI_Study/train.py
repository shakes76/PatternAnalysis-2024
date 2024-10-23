import torch
import torch.optim as optim
import torch.nn as nn
from modules import UNet
from dataset import load_data_2D
import matplotlib.pyplot as plt
import os

# Hyperparameters
epochs = 25
learning_rate = 0.001
batch_size = 8

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = UNet(in_channels=1, out_channels=1).to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_images_filenames = [('keras_slices_seg_train/' + f) for f in os.listdir('keras_slices_seg_train') if f.endswith('.nii.gz')]
train_labels_filenames = [('keras_slices_train/' + f) for f in os.listdir('keras_slices_train') if f.endswith('.nii.gz')]
val_images_filenames = [('keras_slices_seg_validate/' + f) for f in os.listdir('keras_slices_seg_validate') if f.endswith('.nii.gz')]
val_labels_filenames = [('keras_slices_validate/' + f) for f in os.listdir('keras_slices_validate') if f.endswith('.nii.gz')]
train_images = load_data_2D(train_images_filenames)
train_labels = load_data_2D(train_labels_filenames)
val_images = load_data_2D(val_images_filenames)
val_labels = load_data_2D(val_labels_filenames)

# Training loop
def train_model():
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        for i in range(len(train_images)):
            image = train_images[i].to(device)
            label = train_labels[i].to(device)

            # Forward pass
            outputs = model(image)
            loss = criterion(outputs, label)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_images))

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for i in range(len(val_images)):
                image = val_images[i].to(device)
                label = val_labels[i].to(device)
                outputs = model(image)
                loss = criterion(outputs, label)
                epoch_val_loss += loss.item()

        val_losses.append(epoch_val_loss / len(val_images))

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    # Plot losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    torch.save(model.state_dict(), 'unet_model.pth')


train_model()
