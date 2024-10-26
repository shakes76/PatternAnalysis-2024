# PREDICT.PY

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import UNet
from dataset import SegmentationData  # Ensure this is the path to your SegmentationData class
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

def dice_coefficient(pred, target, epsilon=1e-6):
    """
    Computes the Dice Coefficient for each class.

    Parameters:
        pred (torch.Tensor): Predicted segmentation mask (batch_size, n_classes, H, W).
        target (torch.Tensor): Ground truth segmentation mask (batch_size, H, W).
        epsilon (float): Small value to avoid division by zero.

    Returns:
        dice (numpy.ndarray): Dice coefficient for each class.
    """
    num_classes = pred.shape[1]
    dice = np.zeros(num_classes)
    pred = pred.argmax(dim=1)  # Convert logits to class indices

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        pred_sum = pred_inds.sum().item()
        target_sum = target_inds.sum().item()

        dice_cls = (2. * intersection + epsilon) / (pred_sum + target_sum + epsilon)
        dice[cls] = dice_cls

    return dice

# Datasets directories
test_image_dir = './data/HipMRI_study_keras_slices_data/test'
test_label_dir = './data/HipMRI_study_keras_slices_data/seg_test'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Test Dataset
print("Loading Test Data")
test_dataset = SegmentationData(
    test_image_dir, test_label_dir,
    norm_image=False, categorical=True, dtype=torch.float32
)

# Create dataloader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize the model
n_channels = 1  # Assuming input images are grayscale
n_classes = test_dataset.num_classes  # Number of classes in the dataset
model = UNet(n_channels=n_channels, n_classes=n_classes)
model.to(device)

# Load the latest model checkpoint
# Find the latest epoch number from saved model files
model_files = [f for f in os.listdir('.') if f.startswith('model_epoch_') and f.endswith('.pth')]
if not model_files:
    raise FileNotFoundError("No model checkpoint files found.")
latest_model_file = max(model_files, key=lambda x: int(x.split('_')[-1].split('.pth')[0]))
print(f"Loading model from checkpoint: {latest_model_file}")
model.load_state_dict(torch.load(latest_model_file, map_location=device))

model.eval()
dice_scores = []

with torch.no_grad():
    print("Running inference on test data...")
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)  # Shape: (batch_size, n_channels, H, W)
        labels = labels.to(device)  # Shape: (batch_size, n_classes, H, W)
        labels_cls = labels.argmax(dim=1)  # Convert one-hot labels to class indices

        # Forward pass
        outputs = model(images)  # Shape: (batch_size, n_classes, H, W)

        # Compute Dice coefficient
        dice = dice_coefficient(outputs, labels_cls)
        dice_scores.append(dice)

# Convert list of dice_scores to a numpy array
dice_scores = np.array(dice_scores)  # Shape: (num_samples, num_classes)

# Compute mean Dice coefficient for each class
mean_dice = dice_scores.mean(axis=0)

# Print Dice scores for each class
print("\nDice Similarity Coefficient for each class:")
for cls_idx, dice in enumerate(mean_dice):
    print(f"Class {cls_idx}: Dice = {dice:.4f}")
