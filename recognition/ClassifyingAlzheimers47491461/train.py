import random

from modules import *
from dataset import *
import torch.nn as nn
import torch.optim as optim
import torch.fft
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

def train_model():
    # Instantiate GFNet model with default parameters as defined in paper
    model = GFNet(
        img_size=224, patch_size=14, in_chans=1, num_classes=2, embed_dim=256, depth=12,
        mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1).to(device)
    print("Model loaded")

    # Gets the dataset
    dataset = process(colab=False)[0]
    print("Data processed")

    # Splits dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Separate data-loading for validation and training sets
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler
    num_epochs = 50
    # Warmup as per paper
    warmup_epochs = 5
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    # Define a custom learning rate schedule function (as per paper)
    def lr_lambda(current_step):
        # During the warm-up phase, linearly increase the learning rate
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # After warm-up, apply cosine decay to the learning rate
        # Cosine decay starts at the highest learning rate (1.0) and gradually decreases
        # until it reaches 0.5 * (1 + cos(pi)) = 0 at the end of training
        return max(
            0.0, 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
        )

    # Initialize the scheduler with the lambda function controlling the learning rate schedule
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize lists to track training and validation losses over epochs
    global_step = 0
    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    # Main training loop over epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Iterate over training data batches
        for images, labels in train_loader:
            # Move data to the GPU
            images, labels = images.to(device), labels.to(device)
            # Zero out previous gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Calculate the loss for the batch
            loss = criterion(outputs, labels)

            loss.backward()  # Backpropagation: compute gradients of the loss w.r.t. model parameters
            optimizer.step()  # Update model parameters based on gradients
            scheduler.step()  # Adjust learning rate according to scheduler

            running_loss += loss.item()  # Accumulate batch loss
            global_step += 1  # Track total steps

        # Calculate and store average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation step (no gradient updates)
        # Set model to evaluation mode
        model.eval()
        # Initialize running validation loss
        val_running_loss = 0.0
        # Disable gradient computation for validation
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                # Model predictions for validation data
                val_outputs = model(val_images)
                # Calculate validation loss
                val_loss = criterion(val_outputs, val_labels)
                # Accumulate validation loss
                val_running_loss += val_loss.item()

        # Calculate and store average validation loss for the epoch
        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        # Record epoch number for plotting
        epoch_list.append(epoch + 1)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save the trained model state
    model_save_path = '/content/drive/MyDrive/model_state.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # Return recorded losses and epoch data
    return train_loss_list, val_loss_list, epoch_list

# Device configuration (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training and evaluation script
if __name__ == "__main__":
    train_loss_list, val_loss_list, epoch_list = train_model()
    print(f"Training Losses: {train_loss_list}")
    print(f"Validation Losses: {val_loss_list}")
    print(f"Epochs: {epoch_list}")

    # Plot training and validation loss over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_loss_list, label='Training Loss')
    plt.plot(epoch_list, val_loss_list, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

