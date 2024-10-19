import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import UNet
from tqdm import tqdm
import numpy as np
import random
from typing import Tuple, Dict

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Dice Loss
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

def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1.) -> float:
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + smooth)
    return dice.mean().item()

def initialize_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Initialize training and validation dataloaders."""
    train_dataset = HipMRIDataset(data_dir=config['data_dir'], seg_dir=config['seg_dir'])
    val_dataset = HipMRIDataset(data_dir=config['val_data_dir'], seg_dir=config['val_seg_dir'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    return train_loader, val_loader

def initialize_model(config: Dict, device: torch.device) -> nn.Module:
    """Initialize the UNet model and move it to the specified device."""
    model = UNet(
        n_channels=config['n_channels'], 
        n_classes=config['n_classes'], 
        bilinear=config['bilinear'],
        base_filters=config['base_filters'],
        use_batchnorm=config['use_batchnorm']
    ).to(device)
    return model

def initialize_loss_function() -> nn.Module:
    """Initialize the combined loss function (BCE + Dice)."""
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    
    class CombinedLoss(nn.Module):
        def __init__(self, bce: nn.Module, dice: nn.Module, alpha: float = 0.5):
            super(CombinedLoss, self).__init__()
            self.bce = bce
            self.dice = dice
            self.alpha = alpha
        
        def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return self.alpha * self.bce(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)
    
    return CombinedLoss(bce_loss, dice_loss, alpha=0.5)

def initialize_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Initialize the optimizer."""
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    return optimizer

def initialize_scheduler(optimizer: optim.Optimizer, config: Dict) -> optim.lr_scheduler.ReduceLROnPlateau:
    """Initialize the learning rate scheduler."""
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config['lr_factor'], 
        patience=config['lr_patience'], 
        verbose=True
    )
    return scheduler

def train_model(model: nn.Module, 
                device: torch.device, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                scheduler: optim.lr_scheduler.ReduceLROnPlateau, 
                num_epochs: int, 
                save_path: str) -> None:
    """Train the UNet model."""
    best_dice = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        train_dice = 0.0
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch') as pbar:
            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                train_dice += dice_coefficient(outputs, masks)
                
                pbar.set_postfix({'loss': loss.item(), 'dice': train_dice / (pbar.n + 1)})
                pbar.update(1)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = train_dice / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}, Dice Coefficient: {avg_val_dice:.4f}")
        
        # Step the scheduler
        scheduler.step(avg_val_loss)
        
        # Checkpoint
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Best model saved with Dice Coefficient: {best_dice:.4f}")
    
    print(f"Training complete. Best Dice Coefficient: {best_dice:.4f}")