import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import HipMRIDataset
from modules import UNet
from tqdm import tqdm
import numpy as np
import random
from typing import Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
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

# Dice Coefficient Metric
def dice_coefficient(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1.) -> float:
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + smooth)
    return dice.mean().item()

# Combined BCE + Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, weight=None):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)
        self.dice = DiceLoss()
        self.alpha = alpha
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.bce(preds, targets) + (1 - self.alpha) * self.dice(preds, targets)


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
    
    def __call__(self, score, model, save_path):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    
    def save_checkpoint(self, model, save_path):
        torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        if self.verbose:
            print('Validation score improved. Saving model.')

# Initialize DataLoaders
def initialize_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Initialize training and validation dataloaders."""
    train_dataset = HipMRIDataset(
        data_dir=config['data_dir'], 
        seg_dir=config['seg_dir'], 
        transform=config.get('transform_train'),
        categorical=config.get('categorical', False)
    )
    val_dataset = HipMRIDataset(
        data_dir=config['val_data_dir'], 
        seg_dir=config['val_seg_dir'], 
        transform=config.get('transform_val'),
        categorical=config.get('categorical', False)
    )
    
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

# Initialize Model
def initialize_model(config: Dict, device: torch.device) -> nn.Module:
    """Initialize the UNet model and move it to the specified device."""
    model = UNet(
        n_channels=config['n_channels'], 
        n_classes=config['n_classes'], 
        bilinear=config['bilinear'],
        base_filters=config['base_filters'],
        use_batchnorm=config['use_batchnorm']
    ).to(device)
    
    # Apply Xavier Initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model

# Initialize Loss Function
def initialize_loss_function() -> nn.Module:
    """Initialize the combined loss function (BCE + Dice)."""
    return CombinedLoss(alpha=0.5)

# Initialize Optimizer
def initialize_optimizer(model: nn.Module, config: Dict) -> optim.Optimizer:
    """Initialize the optimizer."""
    if config.get('optimizer', 'Adam') == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    return optimizer

# Initialize Learning Rate Scheduler
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

# Training Loop
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
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(1, num_epochs + 1):
        # Training Phase
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
                
                # Gradient Inspection
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            print(f"NaN gradients in {name}")
                        if torch.isinf(param.grad).any():
                            print(f"Inf gradients in {name}")
                    else:
                        print(f"No gradients for {name}")
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                train_dice += dice_coefficient(outputs, masks)
                
                pbar.set_postfix({'loss': loss.item(), 'dice': train_dice / (pbar.n + 1)})
                pbar.update(1)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_dice = train_dice / len(train_loader)
        print(f"Epoch {epoch} Training Loss: {avg_loss:.4f}, Dice Coefficient: {avg_dice:.4f}")
        
        # Validation Phase
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
        
        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)
        
        # Early Stopping Check
        early_stopping(avg_val_dice, model, save_path)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
        # Checkpointing
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"Best model saved with Dice Coefficient: {best_dice:.4f}")
        
        # Periodic Checkpoints
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
    
    print(f"Training complete. Best Dice Coefficient: {best_dice:.4f}")


# Visualization Function
def visualize_predictions(model: nn.Module, 
                          dataset: HipMRIDataset, 
                          device: torch.device, 
                          num_samples: int = 5, 
                          save_dir: str = 'visualizations') -> None:
    """
    Visualize model predictions alongside ground truth masks.
    
    Args:
        model (nn.Module): Trained model.
        dataset (HipMRIDataset): Dataset to sample from.
        device (torch.device): Device to perform computations on.
        num_samples (int): Number of samples to visualize.
        save_dir (str): Directory to save visualization images.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    for i in range(num_samples):
        image, mask = dataset[i]
        image = image.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            preds = torch.sigmoid(output)
            preds = (preds > 0.5).float()
        
        image_np = image.cpu().numpy().squeeze()
        mask_np = mask.cpu().numpy().squeeze()
        preds_np = preds.cpu().numpy().squeeze()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_np, cmap='gray')
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap='gray')
        plt.title('Ground Truth Mask')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(preds_np, cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')
        
        plt.savefig(os.path.join(save_dir, f'sample_{i}.png'))
        plt.close()
        print(f"Saved visualization for sample {i} as 'sample_{i}.png'")

def create_small_subset(dataset: HipMRIDataset, num_samples: int = 5) -> Subset:
    indices = list(range(num_samples))
    return Subset(dataset, indices)

# Main Function
def main():
    set_seed(42)
    
    # Configuration Dictionary
    config = {
        'data_dir': 'keras_slices_train',  
        'seg_dir': 'keras_slices_seg_train',  
        'val_data_dir': 'keras_slices_validate',  
        'val_seg_dir': 'keras_slices_seg_validate',  
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 5,
        'num_workers': 4,
        'n_channels': 1,
        'n_classes': 1,
        'bilinear': True,
        'base_filters': 64,
        'use_batchnorm': True,
        'save_path': 'checkpoints',
        'optimizer': 'Adam',  # or 'SGD'
        'categorical': False,  # Set to True if dealing with multi-class masks
        'transform_train': A.Compose([
            A.Resize(height=256, width=256),  # Ensure all images are 256x256
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(mean=(0.0,), std=(1.0,)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}),
        'transform_val': A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=(0.0,), std=(1.0,)),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'}),
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs(config['save_path'], exist_ok=True)
    
    # Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize DataLoaders, Model, Loss, Optimizer, Scheduler
    try:
        full_train_loader, full_val_loader = initialize_dataloaders(config)
        model = initialize_model(config, device)
        criterion = initialize_loss_function()
        optimizer = initialize_optimizer(model, config)
        scheduler = initialize_scheduler(optimizer, config)
    except Exception as e:
        print(f"Error during initialization: {e}")
        return
    
    # Create small subset loaders
    small_train_dataset = create_small_subset(full_train_loader.dataset, num_samples=5)
    small_val_dataset = create_small_subset(full_val_loader.dataset, num_samples=2)
    
    small_train_loader = DataLoader(
        small_train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    small_val_loader = DataLoader(
        small_val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Train on the small subset
    try:
        train_model(model, device, small_train_loader, small_val_loader, criterion, optimizer, scheduler, num_epochs=10, save_path=config['save_path'])
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Visualize Predictions
    visualize_predictions(model, small_val_loader.dataset, device, num_samples=2, save_dir='small_subset_visualizations')
if __name__ == '__main__':
    main()
