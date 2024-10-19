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