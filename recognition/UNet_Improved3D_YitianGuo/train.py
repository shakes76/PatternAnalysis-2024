import os
import glob
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRIDataset, train_transforms, val_transforms  # Ensure these are correctly defined in dataset.py
from modules import UNet3D  # Ensure UNet3D is defined in modules.py
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import nibabel as nib
import csv

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='3D UNet Training Script')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--loss', type=str, default='combined', choices=['dice', 'ce', 'combined'])
parser.add_argument('--dataset_root', type=str, default='/home/groups/comp3710/HipMRI_Study_open')
args = parser.parse_args()

def split_data(dataset_root, seed=42, train_size=0.6, val_size=0.2, test_size=0.2):
    image_paths = sorted(glob.glob(os.path.join(dataset_root, 'semantic_MRs', '*.nii.gz')))
    label_paths = sorted(glob.glob(os.path.join(dataset_root, 'semantic_labels_only', '*.nii.gz')))

    assert len(image_paths) == len(label_paths)

    # Split data into training and temporary set
    train_image_paths, temp_image_paths, train_label_paths, temp_label_paths = train_test_split(
        image_paths, label_paths, test_size=(1 - train_size), random_state=seed
    )

    # Split temporary set into validation and test sets
    val_image_paths, test_image_paths, val_label_paths, test_label_paths = train_test_split(
        temp_image_paths, temp_label_paths, test_size=(test_size / (test_size + val_size)), random_state=seed
    )

    return {
        'train': (train_image_paths, train_label_paths),
        'val': (val_image_paths, val_label_paths),
        'test': (test_image_paths, test_label_paths)
    }

splits = split_data(args.dataset_root, seed=seed, train_size=0.6, val_size=0.2, test_size=0.2)
train_image_paths, train_label_paths = splits['train']
val_image_paths, val_label_paths = splits['val']

# Initialize training and validation datasets
train_dataset = MRIDataset(
    image_paths=train_image_paths,
    label_paths=train_label_paths,
    transform=train_transforms,
    norm_image=True,
    dtype=np.float32
)

val_dataset = MRIDataset(
    image_paths=val_image_paths,
    label_paths=val_label_paths,
    transform=val_transforms,
    norm_image=True,
    dtype=np.float32
)

# Load data using DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

def save_best_dice(best_class_dice, file_path='best_class_dice.csv'):
    with open(file_path, mode='w', newline='') as csv_file:
        fieldnames = ['Class', 'Best_Dice_Score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for class_name, dice_score in best_class_dice.items():
            writer.writerow({'Class': class_name, 'Best_Dice_Score': dice_score})

def compute_class_weights(label_paths, num_classes):
    class_counts = np.zeros(num_classes)
    for path in label_paths:
        label = nib.load(path).get_fdata().astype(np.uint8)
        unique, counts = np.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[u] += c
    class_weights = class_counts.max() / class_counts
    return torch.tensor(class_weights, dtype=torch.float32).to(args.device)

# Define DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        # Use Softmax to get the probability of each class
        inputs = F.softmax(inputs, dim=1)
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
        # Calculate Dice loss for each class
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3, 4))
        union = inputs.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        # Exclude background class (index 0)
        dice = dice[:, 1:]
        # Calculate average Dice loss
        dice_loss = 1 - dice.mean()
        return dice_loss

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce, weight_dice):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        # Directly use targets, no need for argmax
        loss_ce = self.ce(inputs, targets.squeeze(1))
        loss_dice = self.dice(inputs, targets)
        return self.weight_ce * loss_ce + self.weight_dice * loss_dice

# Initialize model and move to device
model = UNet3D().to(args.device)

# Compute class weights
num_classes = model.final_conv.out_channels
class_weights = compute_class_weights(train_label_paths, num_classes)

# Define loss function
if args.loss == 'dice':
    criterion = DiceLoss().to(args.device)
elif args.loss == 'ce':
    criterion = nn.CrossEntropyLoss().to(args.device)
elif args.loss == 'combined':
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=1.0).to(args.device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Initialize GradScaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()

# Initialize lists for tracking loss and metrics
train_losses = []
valid_losses = []

# Additional lists for validation
val_losses = []
val_dices = []

train_dice_scores = {f'Class_{i}': [] for i in range(num_classes)}
val_dice_scores = {f'Class_{i}': [] for i in range(num_classes)}

def plot_metrics(train_losses, val_losses, train_dice_scores, val_dice_scores, num_classes, save_path='metrics.png'):
    """Plot training and validation loss as well as Dice coefficient for each class"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(20, 10))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()

    # Plot Dice coefficient for each class
    plt.subplot(2, 1, 2)
    for class_index in range(num_classes):
        class_name = f'Class_{class_index}'
        plt.plot(epochs, train_dice_scores[class_name], label=f'Train Dice {class_name}')
        plt.plot(epochs, val_dice_scores[class_name], linestyle='--', label=f'Val Dice {class_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.title('Dice Coefficient Per Class During Training')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def evaluate(model, val_loader, criterion, device, num_classes):
    model.eval()
    val_loss = 0
    class_dice = np.zeros(num_classes)
    class_counts = np.zeros(num_classes)
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device, dtype=torch.float32)  # [B, 1, D, H, W]
            labels = batch['label'].to(device, dtype=torch.long)  # [B, D, H, W] or [B, 1, D, H, W]

            if labels.dim() == 5 and labels.size(1) == 1:
                labels = labels.squeeze(1)  # Now labels is [B, D, H, W]

            with autocast():
                outputs = model(images)  # [B, num_classes, D, H, W]
                loss = criterion(outputs, labels)

            val_loss += loss.item()

            # Calculate predictions
            preds = torch.argmax(outputs, dim=1)  # [B, D, H, W]

            # Now preds and labels are both [B, D, H, W]
            targets = labels

            # Calculate Dice coefficient for each class
            for class_index in range(num_classes):
                pred_class = (preds == class_index).float()
                target_class = (targets == class_index).float()

                intersection = (pred_class * target_class).sum().item()
                union = pred_class.sum().item() + target_class.sum().item()
                if union == 0:
                    dice = 1.0  # If the denominator is 0, it means there is no such class, set dice to 1
                else:
                    dice = (2. * intersection) / (union + 1e-8)
                class_dice[class_index] += dice
                if union > 0:
                    class_counts[class_index] += 1

    # Calculate average Dice coefficient for each class
    avg_class_dice = class_dice / np.maximum(class_counts, 1)
    avg_loss = val_loss / len(val_loader)
    return avg_loss, avg_class_dice

def main():
    start_time = time.time()
    num_epochs = args.epoch
    save_path = 'best_model.pth'
    num_classes = model.final_conv.out_channels

    # Initialize a dictionary to store the best Dice score for each class
    best_class_dice = {f'Class_{i}': 0.0 for i in range(num_classes)}

    # Early stopping parameters
    patience = 10
    counter = 0

    print("Begin training")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = []
        epoch_train_dice = {f'Class_{i}': [] for i in range(num_classes)}

        for idx, batch in enumerate(train_dataloader):
            images = batch['image'].to(args.device, dtype=torch.float32)
            labels = batch['label'].to(args.device, dtype=torch.long)

            if labels.dim() == 5 and labels.size(1) == 1:
                labels = labels.squeeze(1)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss.append(loss.item())

            # Calculate Dice coefficient for each class
            preds = torch.argmax(outputs, dim=1)
            targets = labels

            for class_index in range(num_classes):
                pred_class = (preds == class_index).float()
                target_class = (targets == class_index).float()

                intersection = (pred_class * target_class).sum().item()
                union = pred_class.sum().item() + target_class.sum().item()
                if union == 0:
                    dice = 1.0
                else:
                    dice = (2. * intersection) / (union + 1e-8)
                epoch_train_dice[f'Class_{class_index}'].append(dice)

            # Print training progress
            if (idx + 1) % max(1, len(train_dataloader) // 2) == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{idx + 1}/{len(train_dataloader)}], "
                    f"Loss: {loss.item():.4f}, Elapsed Time: {elapsed:.2f}s"
                )

        # Record average loss and Dice coefficient at the end of each epoch
        avg_train_loss = np.mean(epoch_train_loss)
        avg_train_dice = {class_name: np.mean(dice_list) for class_name, dice_list in epoch_train_dice.items()}
        train_losses.append(avg_train_loss)
        for class_name, dice in avg_train_dice.items():
            train_dice_scores[class_name].append(dice)

        # Validation phase
        avg_val_loss, avg_val_dice = evaluate(model, val_loader, criterion, args.device, num_classes)
        val_losses.append(avg_val_loss)
        for class_index, dice in enumerate(avg_val_dice):
            class_name = f'Class_{class_index}'
            val_dice_scores[class_name].append(dice)

        # Check if there is an improvement for each class
        improved = False
        for class_index in range(num_classes):
            class_name = f'Class_{class_index}'
            if avg_val_dice[class_index] > best_class_dice[class_name]:
                best_class_dice[class_name] = avg_val_dice[class_index]
                improved = True

        if improved:
            torch.save(model.state_dict(), save_path)
            print(f"Saved the best model at epoch {epoch}")
            print(f"Best Class Dice Scores: {best_class_dice}")
            counter = 0  # Reset early stopping counter
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

        # Print validation results for the current epoch
        print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print("Validation Dice Scores:")
        for class_index, dice in enumerate(avg_val_dice):
            print(f"  Class {class_index}: {dice:.4f}")

    # Plot loss and Dice coefficient
    plot_metrics(train_losses, val_losses, train_dice_scores, val_dice_scores, num_classes,
                 save_path='training_metrics.png')

    # Save the best Dice score for each class
    save_best_dice(best_class_dice)

    print("Finished Training")

if __name__ == '__main__':
    main()
