import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import load_all_data
from modules import UNet2D
import numpy as np

# GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate the Dice coefficient
def dice_coefficient(pred, target, smooth=1.0):
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# DICE loss
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Mixed Loss (Dice Loss + BCELoss.)）
def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

# Validate the model
def validate_model(model, criterion, val_data):
    model.eval()
    val_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for i in range(val_data.shape[0]):
            inputs = torch.tensor(val_data[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            masks = inputs
            masks = torch.clamp(masks, 0, 1).to(device)
            masks = torch.cat([masks, 1 - masks], dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            dice_score = dice_coefficient(outputs, masks)
            dice_scores.append(dice_score.item())

    avg_loss = val_loss / val_data.shape[0]
    avg_dice = np.mean(dice_scores)
    return avg_loss, avg_dice

# Train the model
def train_model(model, criterion, optimizer, scheduler, train_data, val_data, num_epochs=50):
    model.train()
    train_losses = []
    val_losses = []
    val_dice_scores = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f'Starting Epoch {epoch+1} of {num_epochs}')

        # Training loops
        for i in range(train_data.shape[0]):
            inputs = torch.tensor(train_data[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            masks = inputs
            masks = torch.clamp(masks, 0, 1).to(device)
            masks = torch.cat([masks, 1 - masks], dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / train_data.shape[0]
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

        # Validate the model
        val_loss, val_dice = validate_model(model, criterion, val_data)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}')

        # Adjust the learning rate
        scheduler.step(val_loss)

    # Plot training and validation loss curves
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.show()

    return val_dice_scores

# main
def main():
    image_dir = r'C:\Users\舒画\Downloads\HipMRI_study_keras_slices_data\keras_slices_train'
    images = next(load_all_data(image_dir, normImage=True, target_shape=(256, 256), batch_size=32))

    print(f'Loaded {images.shape[0]} images with shape {images.shape[1:]}')

    # The dataset is divided into 80% training set, 10% validation set, and 10% test set
    train_size = int(0.8 * images.shape[0])
    val_size = int(0.1 * images.shape[0])
    test_size = images.shape[0] - train_size - val_size

    train_data = images[:train_size]
    val_data = images[train_size:train_size + val_size]
    test_data = images[train_size + val_size:]

    # Define the model
    model = UNet2D(in_channels=1, out_channels=2).to(device)

    # Define the loss function and optimizer
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Use the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Train the model
    train_model(model, criterion, optimizer, scheduler, train_data, val_data, num_epochs=50)

    # Save the model
    torch.save(model.state_dict(), 'unet_model1.pth')
    print("Model saved as 'unet_model1.pth'")

if __name__ == '__main__':
    main()
