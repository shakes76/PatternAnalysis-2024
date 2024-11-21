import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from modules import Basic3DUNet, DiceLoss, Improved3DUNet, calculate_dice
import time
import os
from dataset import load_data_3D, Custom3DDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from predict import test_and_visualize
import torchvision.transforms as transforms
import numpy as np
import scipy.ndimage
from monai.transforms import RandSpatialCrop

# Model params
num_epochs = 20
learning_rate = 0.001
batch_size = 2
# Check which device to use for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomFlip3D:
    def __init__(self, flip_depth=True, flip_height=True, flip_width=True):
        self.flip_depth = flip_depth
        self.flip_height = flip_height
        self.flip_width = flip_width

    def __call__(self, x):
        # Flip DImensions
        if self.flip_depth and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[2])
        if self.flip_height and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[3])
        if self.flip_width and torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[4])
        return x


class RandomRotate3D:
    def __init__(self, max_angle=30):
        self.max_angle = max_angle

    def __call__(self, img):
        # Convert the angle to radians for compatibility with torch
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        # Choose a random pair of axes for rotation
        axes = [(2, 3), (3, 4), (2, 4)]
        chosen_axes = np.random.choice(len(axes))
        axes_to_rotate = axes[chosen_axes]

        # Rotate the tensor using torch.rot90
        k = int(angle // 90)
        img = torch.rot90(img, k=k, dims=axes_to_rotate)
        return img


# Transformations
transform_train = transforms.Compose([
    RandomFlip3D(flip_depth=True, flip_height=True, flip_width=True),
    RandomRotate3D(max_angle=30),
    # RandSpatialCrop((1, 128, 128, 64), random_size=False),
    transforms.Normalize((0.5,), (0.5,))


])

# Get the dataset and split into train, test, validate
dataset = Custom3DDataset(transform_train)
train_len = int(0.8 * len(dataset))  # fix percentages
val_len = int(0.1*len(dataset))
test_len = len(dataset) - train_len - val_len
train_loader, validate_loader, test_loader = random_split(
    dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_loader, batch_size=batch_size,
                          shuffle=True, num_workers=batch_size)
validate_loader = DataLoader(
    validate_loader, batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_loader, batch_size=batch_size,
                         shuffle=False, num_workers=4)

# initialise Unet and loss/optimizer
model = Improved3DUNet(in_channels=1, out_channels=6).to(device)
weights = [1.0, 1.0, 2.0, 2.0, 12.0, 12.0]
loss_func = DiceLoss(weights=weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

sched_linear_1 = optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.0005, max_lr=learning_rate, step_size_up=15, step_size_down=15, mode='triangular', verbose=False, cycle_momentum=False)
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

    # Iterate through Epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        # Iterate through images and masks
        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            # Apply Loss
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
    class_dice_scores = []
    with torch.no_grad():
        for val_images, val_masks in validate_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)
            dice_scores_per_class = calculate_dice(val_outputs, val_masks)
            class_dice_scores.append(dice_scores_per_class)
    avg_class_dice_scores = torch.mean(
        torch.stack(class_dice_scores), dim=0).cpu().numpy()

    for class_idx, dice_score in enumerate(avg_class_dice_scores):
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Class {class_idx} Dice Score: {dice_score:.4f}")

    avg_dice_score = avg_class_dice_scores.mean()
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Average Validation Dice Score: {avg_dice_score:.4f}")


if __name__ == "__main__":
    train()
    test_and_visualize(model, test_loader)
