import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import UNet
from dataset import ProstateDataset
import matplotlib.pyplot as plt

def dice_coeff(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def train(model, train_loader, val_loader, epochs=20, lr=1e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    train_loss_history, val_dice_history = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        val_dice = 0
        with torch.no_grad():
            for images, masks in val_loader:
                outputs = model(images)
                val_dice += dice_coeff(outputs, masks)

        val_dice /= len(val_loader)
        val_dice_history.append(val_dice.item())

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Dice: {val_dice:.4f}")

    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_dice_history, label='Val Dice')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_dataset = ProstateDataset('image_path.nii', 'mask_path.nii')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    model = UNet()
    train(model, train_loader, None)  # For simplicity, adding validation later
