"""
train.py

Author: Darcy Weedman
Student ID: 45816985
COMP3710 HipMRI 2D UNet project
Semester 2, 2024
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import HipMRIDataset
from modules import UNet
from tqdm import tqdm
import numpy as np
import random
import torch.nn.functional as F
from typing import Tuple, Dict
import albumentations as A
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

# Dice Loss Definition
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

def dice_coeff_multiclass(pred, target, num_classes, epsilon=1e-6):
    """
    Compute Dice Coefficient for multi-class segmentation.
    pred: model output tensor of shape (B, C, H, W) or argmax'd prediction of shape (B, H, W)
    target: ground truth tensor of shape (B, H, W) with class indices
    """
    dice_scores = []
    
    # If pred is not argmax'd yet, we do it here
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)
    
    for class_id in range(num_classes):
        class_pred = (pred == class_id).float()
        class_target = (target == class_id).float()
        intersection = (class_pred * class_target).sum(dim=(1,2))
        union = class_pred.sum(dim=(1,2)) + class_target.sum(dim=(1,2))
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.mean().item())
    return dice_scores

class DiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (pred * target_onehot).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
        dice = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dice.mean()
    
def debug_tensors(tensor_dict):
    for name, tensor in tensor_dict.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logging.error(f"{name} contains NaN or Inf values")
        
        if tensor.dtype in [torch.float32, torch.float64]:
            logging.debug(f"{name} - min: {tensor.min().item():.4f}, max: {tensor.max().item():.4f}, mean: {tensor.mean().item():.4f}")
        else:
            logging.debug(f"{name} - min: {tensor.min().item()}, max: {tensor.max().item()}, unique values: {torch.unique(tensor).tolist()}")

def visualize_predictions(model, loader, device, num_classes, save_dir='predictions'):
    """
    Visualize model predictions alongside ground truth masks for multi-class segmentation.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                img = images[i].cpu().squeeze().numpy()
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img, cmap='gray')
                plt.title('Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='nipy_spectral', vmin=0, vmax=num_classes-1)
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred, cmap='nipy_spectral', vmin=0, vmax=num_classes-1)
                plt.title('Prediction')
                plt.axis('off')

                plt.savefig(os.path.join(save_dir, f'sample_{idx * loader.batch_size + i}.png'))
                plt.close()

                if idx * loader.batch_size + i >= 4:  # Save first 5 samples
                    return

def main():
    set_seed(42)

    # Configuration
    train_image_dir = 'keras_slices_train'
    train_mask_dir = 'keras_slices_seg_train'
    val_split = 0.2
    batch_size = 4
    num_epochs = 50
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Dataset and DataLoader setup
    dataset = HipMRIDataset(
        image_dir=train_image_dir, 
        mask_dir=train_mask_dir, 
        norm=True, 
        target_size=(256, 256)
    )
    num_classes = dataset.num_classes
    dataset_size = len(dataset)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = UNet(n_channels=1, n_classes=num_classes).to(device)
    logging.info("Model initialized.")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    logging.info("Loss, optimizer, and scheduler initialized.")

    best_dice = 0.0
    save_path = 'best_model_simple_unet.pth'

    # Initialize lists to store metrics for plotting
    train_losses = []
    train_dices = []
    val_losses = []
    val_dices = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dice = [0.0] * num_classes
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate losses
            ce_loss = criterion(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = ce_loss + d_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            batch_dice = dice_coeff_multiclass(outputs, masks, num_classes)
            for i in range(num_classes):
                epoch_dice[i] += batch_dice[i]

            loop.set_postfix(ce_loss=ce_loss.item(), dice_loss=d_loss.item(), total_loss=loss.item(), dice=np.mean(batch_dice))

        avg_loss = epoch_loss / len(train_loader)
        avg_dice = [d / len(train_loader) for d in epoch_dice]
        train_losses.append(avg_loss)
        train_dices.append(avg_dice)
        logging.info(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Dice={avg_dice}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = [0.0] * num_classes
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                ce_loss = criterion(outputs, masks)
                d_loss = dice_loss(outputs, masks)
                loss = ce_loss + d_loss
                val_loss += loss.item()
                batch_dice = dice_coeff_multiclass(outputs, masks, num_classes)
                for i in range(num_classes):
                    val_dice[i] += batch_dice[i]


        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = [d / len(val_loader) for d in val_dice]
        val_losses.append(avg_val_loss)
        val_dices.append(avg_val_dice)
        logging.info(f"Validation: Loss={avg_val_loss:.4f}, Dice={avg_val_dice}")

        # Learning rate scheduling
        scheduler.step(avg_val_dice[1])  # Assuming prostate is class 1

        # Save Best Model
        if avg_val_dice[1] > best_dice:  # Assuming prostate is class 1
            best_dice = avg_val_dice[1]
            torch.save(model.state_dict(), save_path)
            logging.info(f"Best model saved with Prostate Dice={best_dice:.4f}")

    logging.info("Training complete.")

    # Plot Training and Validation Metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot([dice[1] for dice in train_dices], label='Train Prostate Dice')  # Assuming prostate is class 1
    plt.plot([dice[1] for dice in val_dices], label='Validation Prostate Dice')  # Assuming prostate is class 1
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Prostate Dice Coefficient over Epochs')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # Load Best Model for Visualization
    if best_dice > 0:
        model.load_state_dict(torch.load(save_path))
        visualize_predictions(model, val_loader, device, num_classes)
        logging.info("Sample predictions saved.")
    else:
        logging.warning("No improvement during training. Best model not saved.")

if __name__ == "__main__":
    main()