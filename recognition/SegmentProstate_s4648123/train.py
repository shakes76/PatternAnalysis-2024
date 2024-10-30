import time

import numpy as np
import torch

from dataset import get_dataloaders

from modules import UNet3D
from utils import plot_and_save
from config import MODEL_PATH

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Dice(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the Dice loss module.

        Args:
            smooth (float): Smoothing factor to avoid division by zero when calculating the Dice coefficient.
        """
        super(Dice, self).__init__()
        self.smooth = smooth

    def dice(self, pred, target):
        """
        Compute the Dice coefficient between 3D predictions and targets.

        Args:
            pred (torch.Tensor): Model predictions with shape (batch_size, classes/channels, depth, height, width).
            target (torch.Tensor): Ground truth with shape (batch_size, 1, depth, height, width).

        Returns:
            torch.Tensor: Dice coefficients for each class.
        """
        # Apply softmax to logits to get probabilities
        pred = torch.softmax(pred, dim=1)  # (B, C, D, H, W)

        # Define the axes for reduction (batch, depth, height, width)
        reduce_axis = [0] + list(range(2, len(pred.shape)))  # [0, 2, 3, 4]

        # Compute the intersection and union for each class
        intersection = torch.sum(pred * target, dim=reduce_axis)  # (num_classes,)
        ground_o = torch.sum(target, dim=reduce_axis)  # (num_classes,)
        pred_o = torch.sum(pred, dim=reduce_axis)  # (num_classes,)

        # Compute the denominator for Dice coefficient
        denominator = ground_o + pred_o

        # Compute Dice coefficient for each class
        f = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return f

    def forward(self, logits, target):
        """
        Compute the Dice loss.

        Args:
            logits (Tensor): Model outputs with shape (batch_size, num_classes, height, width).
            target (Tensor): Ground truth with shape (batch_size, 1, height, width).

        Returns:
            Tensor: Dice loss.
        """
        coeff = self.dice(logits, target)  # Compute Dice coefficient
        dice_loss = 1 - torch.mean(coeff)  # Mean over classes

        return dice_loss


def train(model, dataloader, optimizer, crit, early_stop=False):
    model.train()
    epoch_loss = 0
    torch.manual_seed(2809)  # reproducibility
    # Determine the number of batches to process
    num_batches = 4 if early_stop else len(dataloader)

    for i, batch_data in enumerate(dataloader):
        if i >= num_batches:  # Stop if we have processed the desired number of batches
            break

        torch.cuda.empty_cache()
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = crit(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss += loss.item()  # Accumulate loss

    epoch_loss = epoch_loss / num_batches
    return epoch_loss


def validate(model, dataloader, crit):
    model.eval()  # Set model to evaluation mode
    dice_scores = []

    with torch.no_grad():  # Disable gradient computation
        torch.manual_seed(2809)  # reproducibility
        for batch_data in dataloader:
            torch.cuda.empty_cache()
            images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            pred = model(images)  # Forward pass
            new_dice_score = crit.dice(pred, labels)
            dice_scores.append(new_dice_score.cpu().numpy())

    dice_scores = np.mean(dice_scores, axis=0)
    return dice_scores

if __name__ == '__main__':
    """
    Main function to run the training and validation processes.
    """

    # Set up datasets and DataLoaders
    batch_size = 8
    train_loader, val_loader, test_loader = get_dataloaders(train_batch=batch_size, val_batch=batch_size)

    # Initialize model
    unet = UNet3D(n_channels=1, n_classes=6)
    unet = unet.to(device)

    epochs = 15
    criterion = Dice()
    optimizer = torch.optim.Adam(unet.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_metric = float(0.)
    best_state = unet.state_dict()

    train_losses, val_losses = [], []
    dice_scores_per_class = [[] for _ in range(5)]
    lrs, weight_decays = [], []

    early_stop=True
    train_start_time = time.time()

    # Training and evaluation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(unet, train_loader, optimizer, criterion, early_stop)
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}")

        dice_scores = validate(unet, val_loader, criterion)
        val_loss = 1.0 - np.mean(dice_scores)
        val_losses.append(val_loss)

        print(f"Validation Loss: {val_loss:.4f}, Dice Scores: {dice_scores}")

        for i, score in enumerate(dice_scores):
            dice_scores_per_class[i].append(score)

        lrs.append([pg['lr'] for pg in optimizer.param_groups])
        weight_decays.append([pg['weight_decay'] for pg in optimizer.param_groups])
        scheduler.step()

        if val_loss < best_metric:
            best_metric = val_loss
            best_state = unet.state_dict()
            torch.save(best_state, MODEL_PATH)

    train_time = time.time() - train_start_time  # Calculate elapsed time
    print(f"Total training time: {train_time:.2f} seconds")

    # Prepare x-axis values
    epochs_range = range(1, epochs + 1)

    # Plot (1) Train and validation loss vs epochs
    plot_and_save(epochs_range, [train_losses, val_losses], ["Train Loss", "Validation Loss"],
        "Train and Validation Loss", "Epochs", "Loss", "train_val_loss.png")

    # Plot (2) Dice score of each class vs epochs
    plot_and_save(epochs_range, dice_scores_per_class, [f"Class {i}" for i in range(5)],
        "Dice Score per Class", "Epochs", "Dice Score", "dice_scores.png")

    # Plot (3) Learning rate and weight decay vs epochs
    lr_values = [lr[0] for lr in lrs]
    wd_values = [wd[0] for wd in weight_decays]
    plot_and_save(epochs_range, [lr_values, wd_values], ["Learning Rate", "Weight Decay"],
        "Learning Rate and Weight Decay", "Epochs", "Value", "lr_wd.png")

