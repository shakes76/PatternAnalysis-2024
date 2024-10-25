import torch
import torch.optim as optim
import torch.nn as nn
from modules import UNet, dice_coefficient
from dataset import load_data_2D
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
epochs = 25
learning_rate = 0.001
batch_size = 16

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = UNet(in_channels=1, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_images_filenames = [('keras_slices_seg_train/' + f) for f in os.listdir('keras_slices_seg_train') if f.endswith('.nii.gz')]
train_labels_filenames = [('keras_slices_train/' + f) for f in os.listdir('keras_slices_train') if f.endswith('.nii.gz')]
val_images_filenames = [('keras_slices_seg_validate/' + f) for f in os.listdir('keras_slices_seg_validate') if f.endswith('.nii.gz')]
val_labels_filenames = [('keras_slices_validate/' + f) for f in os.listdir('keras_slices_validate') if f.endswith('.nii.gz')]
train_images = load_data_2D(train_images_filenames, normImage=True)
train_labels = load_data_2D(train_labels_filenames, normImage=True)
val_images = load_data_2D(val_images_filenames, normImage = True)
val_labels = load_data_2D(val_labels_filenames, normImage = True)
# Training loop
train_images = torch.tensor(train_images).float().unsqueeze(1)
train_labels = torch.tensor(train_labels).float().unsqueeze(1)

def train_model():
    train_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        epoch_val_loss = 0
        for i in range(0, len(train_images), batch_size):
            # Batch Processing
            batch_images = train_images[i:i + batch_size].to(device)
            batch_labels = train_labels[i:i + batch_size].to(device)
            batch_val_images = val_images[i:i + batch_size]
            batch_val_labels = val_labels[i:i + batch_size]
            batch_images = np.expand_dims(batch_images, axis=1).astype(np.float16)
            batch_labels = np.expand_dims(batch_labels, axis=1).astype(np.float16)
            batch_val_images = np.expand_dims(batch_val_images, axis=1).astype(np.float16)
            batch_val_labels = np.expand_dims(batch_val_labels, axis=1).astype(np.float16)
            batch_images = torch.from_numpy(batch_images).float().to(device)
            batch_labels = torch.from_numpy(batch_labels).float().to(device)
            batch_val_images = torch.from_numpy(batch_val_images).float().to(device)
            batch_val_labels = torch.from_numpy(batch_val_labels).float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            val_output = model(batch_val_images)
            loss = criterion(outputs, batch_labels)
            val_loss = criterion(outputs, batch_val_labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            epoch_val_loss += val_loss.item()
        avg_loses = (epoch_train_loss / len(train_images))
        train_losses.append(avg_loses)
        avg_val_losses = (epoch_val_loss / len(val_images))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loses:.4f}, val Loss: {avg_val_losses:.4f}")
    # Plot loss history
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    return model

train_model()

