import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import Basic3DUNet, DiceLoss
import time
import os
from dataset import load_data_3D, Custom3DDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


num_epochs = 5
learning_rate = 0.01
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DataLoader(Custom3DDataset())
train_len = int(0.2 * len(dataset))  # fix percentages
val_len = int(0.7*len(dataset))
test_len = len(dataset) - train_len - val_len
train_loader, validate_loader, test_loader = random_split(
    dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_loader)
validate_loader = DataLoader(validate_loader)
test_loader = DataLoader(test_loader)


model = Basic3DUNet(in_channels=1, out_channels=4).to(device)
loss_func = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

sched_linear_1 = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.0005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode='triangular', verbose=False)
sched_linear_2 = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=min(1, max(0, 0.1)), end_factor=0.001, verbose=False)
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[sched_linear_1, sched_linear_2], milestones=[30])
if not os.path.exists('models'):
    os.makedirs('models')

# Training function


def train():
    total_step = len(train_loader)
    print("> Training")
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.5f}")

        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.5f}")

        # Update learning rate
        scheduler.step()

        # Validation step
        validate(epoch)

        # Save model checkpoint
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pth")

    end = time.time()
    elapsed = end - start
    print("Training took {:.2f} secs or {:.2f} mins in total.".format(
        elapsed, elapsed/60))

# Validation function


def validate(epoch):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for val_images, val_masks in validate_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            dice_score = calculate_dice(val_outputs, val_masks)
            dice_scores.append(dice_score)

    avg_dice_score = sum(dice_scores) / len(dice_scores)
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Validation Dice Score: {avg_dice_score:.4f}")

# Dice Score calculation (for validation)


def calculate_dice(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)  # Convert predictions to binary mask
    # Calculate intersection
    intersection = (y_pred * y_true).sum(dim=(2, 3, 4))
    union = y_pred.sum(dim=(2, 3, 4)) + y_true.sum(dim=(2, 3, 4))
    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    return dice.mean().item()

# Testing function with visualization


def test_and_visualize():
    model.eval()
    with torch.no_grad():
        for test_images, test_masks in test_loader:
            test_images, test_masks = test_images.to(
                device), test_masks.to(device)
            test_outputs = model(test_images)
            predicted_masks = torch.argmax(test_outputs, dim=1)

            # Visualize the first example in the batch
            plt.figure(figsize=(12, 6))

            # Input image
            plt.subplot(1, 3, 1)
            plt.imshow(test_images[0].cpu().squeeze(), cmap='gray')
            plt.title("Input Image")

            # Ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(test_masks[0].cpu().squeeze(), cmap='gray')
            plt.title("Ground Truth Mask")

            # Predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(predicted_masks[0].cpu().squeeze(), cmap='gray')
            plt.title("Predicted Mask")

            plt.tight_layout()
            plt.show()
            break


if __name__ == "__main__":
    train()
    test_and_visualize()
