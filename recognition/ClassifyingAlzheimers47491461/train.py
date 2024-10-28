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
    # Instantiate GFNet model
    model = GFNet(
        img_size=224, patch_size=14, in_chans=1, num_classes=2, embed_dim=256, depth=12,
        mlp_ratio=4., drop_rate=0.1, drop_path_rate=0.1).to(device)
    print("Model loaded")

    # Get the dataset
    dataset = process(colab=False)
    print("Data processed")

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    # Learning rate scheduler
    num_epochs = 1 #50
    warmup_epochs = 0 #5
    total_steps = num_epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    global_step = 0
    train_loss_list = []
    val_loss_list = []
    epoch_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

        avg_train_loss = running_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # Validation step
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        epoch_list.append(epoch + 1)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    model_save_path = '/content/drive/MyDrive/model_state.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    return train_loss_list, val_loss_list, epoch_list

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_loss_list, val_loss_list, epoch_list = train_model()
    print(f"Training Losses: {train_loss_list}")
    print(f"Validation Losses: {val_loss_list}")
    print(f"Epochs: {epoch_list}")

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, train_loss_list, label='Training Loss')
    plt.plot(epoch_list, val_loss_list, label='Validation Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
