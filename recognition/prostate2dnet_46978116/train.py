# train.py

import os
import glob
import torch
import torch.nn as nn
from torch.optim import Adam
from modules import UNet
from dataset import ProstateDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def dice_coefficient(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)  # Convert logits to probabilities
    preds = (preds > threshold).float()  # Apply threshold to get binary predictions
    targets = targets.float()
    
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 5
    batch_size = 8
    learning_rate = 1e-4

    # Training data directories
    train_image_dir = 'keras_slices_data/keras_slices_train'
    train_mask_dir = 'keras_slices_data/keras_slices_seg_train'

    # Validation data directories
    val_image_dir = 'keras_slices_data/keras_slices_validate'
    val_mask_dir = 'keras_slices_data/keras_slices_seg_validate'

    # Adjust glob patterns in train.py

    train_image_paths = sorted(glob.glob(os.path.join(train_image_dir, '*.nii.gz')))
    train_mask_paths = sorted(glob.glob(os.path.join(train_mask_dir, '*.nii.gz')))

    val_image_paths = sorted(glob.glob(os.path.join(val_image_dir, '*.nii.gz')))
    val_mask_paths = sorted(glob.glob(os.path.join(val_mask_dir, '*.nii.gz')))

    # Training dataset
    train_dataset = ProstateDataset(train_image_paths, train_mask_paths, norm_image=True)

    # Validation dataset
    val_dataset = ProstateDataset(val_image_paths, val_mask_paths, norm_image=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    model = UNet(num_classes=1, in_channels=1)
    model = model.to(device)  # Move model to GPU if available

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_dice = 0

        for images, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            dice = dice_coefficient(outputs, masks)
            train_dice += dice.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_dice = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                dice = dice_coefficient(outputs, masks)
                val_dice += dice.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} "
              f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}")

    torch.save(model.state_dict(), 'unet_prostate_segmentation.pth')
