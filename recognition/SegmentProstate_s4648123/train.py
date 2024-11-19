"""
    File name: train.py
    Author: Elliana Rogaczewski-Novak
    Date last modified: 1/11/2024

    Description:  This file contains the training and validation pipeline for the segmentation task, including
    model initialization, loss computation, and performance evaluation using the Dice coefficient. It saves the best
    model to best_unet.pth (defined in config.py)
"""

import time

import numpy as np
import torch

from dataset import get_dataloaders
from modules import UNet3D
from utils import plot_and_save
from config import MODEL_PATH, VISUALISE_RESULTS
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda:0")


class Dice(torch.nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        """
        Initialize the Dice loss module.

        Args:
            smooth (float): Smoothing factor to avoid division by zero when calculating the Dice coefficient.
        """
        super(Dice, self).__init__()
        self.smooth = smooth

    def dice_scores_per_class(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice coefficient for each class.

        Args:
            pred (torch.Tensor): Model predictions with shape (batch_size, classes, height, width, depth).
            target (torch.Tensor): Ground truth with shape (batch_size, classes, height, width, depth).

        Returns:
            torch.Tensor: Dice coefficients for each class.
        """
        # Apply softmax to logits to get probabilities
        pred = torch.softmax(pred, dim=1)  # (B, C, H, W, D)

        # Define the axes for reduction (batch, height, width, depth)
        reduce_axis = [0] + list(range(2, len(pred.shape)))  # [0, 2, 3, 4]

        # Compute the intersection and union for each class
        intersection = torch.sum(pred * target, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(pred, dim=reduce_axis)

        # Compute Dice coefficient for each class excluding background
        f = (2.0 * intersection + self.smooth) / (ground_o + pred_o + self.smooth)

        return f

    def calculate_loss(self, class_dice_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Dice loss from class Dice scores.

        Args:
            class_dice_scores (torch.Tensor): Dice scores for each class.

        Returns:
            torch.Tensor: Calculated Dice loss.
        """
        return 1 - torch.mean(class_dice_scores)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Dice loss.

        Args:
            pred (torch.Tensor): Model outputs with shape (batch_size, num_classes, height, width, depth).
            target (torch.Tensor): Ground truth with shape (batch_size, num_classes, height, width, depth).

        Returns:
            torch.Tensor: Dice loss.
        """
        dice_scores = self.dice_scores_per_class(pred, target)
        return self.calculate_loss(dice_scores)


def train(model, dataloader, optimizer, crit, accumulation_steps = 1) -> float:
    """
    Train the model for one epoch.

    Args:
        model (UNet3D): The model to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for weight updates.
        crit (Dice): Loss function to compute the loss.
        accumulation_steps (int): Number of batches to accumulate gradients over.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0

    for i, batch_data in enumerate(dataloader):
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        # Forward pass
        outputs = model(images)
        loss = crit(outputs, labels)  # Compute loss

        # Scale the loss for accumulation
        loss = loss / accumulation_steps
        loss.backward()

        # Update the weights every 'accumulation_steps' batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

    # Average the epoch loss over all batches
    epoch_loss = epoch_loss * accumulation_steps / len(dataloader)

    return epoch_loss


def validate(model, dataloader, crit) -> tuple[np.ndarray, float]:
    """
    Validate the model on the validation set.

    Args:
        model (UNet3D): The model to be validated.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        crit (Dice): Loss function to compute the loss.

    Returns:
        tuple[np.ndarray, float]: A tuple containing the mean Dice scores and average Dice loss.
    """
    model.eval()  # Set model to evaluation mode
    dice_scores = []
    dice_losses = []

    with torch.no_grad():  # Disable gradient computation
        for batch_data in dataloader:
            images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            pred = model(images)  # Forward pass

            # Calculate per-class dice scores directly
            new_dice_scores = crit.dice_scores_per_class(pred, labels)
            dice_scores.append(new_dice_scores)  # Keep on GPU

            # Calculate the dice loss directly with raw predictions and labels
            dice_loss = crit.calculate_loss(new_dice_scores)
            dice_losses.append(dice_loss.item())

    # Average dice scores across batches, then convert to numpy
    dice_scores = torch.mean(torch.stack(dice_scores), dim=0).cpu().numpy()
    dice_loss = np.mean(dice_losses)
    return dice_scores, dice_loss


if __name__ == '__main__':
    """
    Main function to run the training and validation processes.
    """
    # Set up DataLoaders
    train_loader, val_loader = get_dataloaders()

    # Initialize model
    unet = UNet3D()
    unet = unet.to(device)

    # Training parameters
    epochs = 17
    lr = 0.01
    accumulation_steps = 2
    criterion = Dice()
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)

    # Tracking best model
    best_metric = float(100.)
    best_state = unet.state_dict()

    # Logging info
    train_losses, val_losses = [], []
    dice_scores_per_class = [[] for _ in range(6)]

    train_start_time = time.time()
    print(f"Training parameters: {epochs} epochs, {lr} initial learning rate, "
          f"{accumulation_steps} gradient accumulation steps")

    # Training and evaluation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train(unet, train_loader, optimizer, criterion, accumulation_steps)
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}")

        dice_scores, val_loss = validate(unet, val_loader, criterion)
        val_losses.append(val_loss)

        print(f"Validation Loss: {val_loss:.4f}, Dice Scores: {[f'{score:.4f}' for score in dice_scores.tolist()]}")

        for i, score in enumerate(dice_scores):
            dice_scores_per_class[i].append(score)

        if val_loss < best_metric:
            best_metric = val_loss
            best_state = unet.state_dict()
            torch.save(best_state, MODEL_PATH)

        scheduler.step()

    train_time = time.time() - train_start_time
    print(f"Total training time: {train_time:.2f} seconds")

    if VISUALISE_RESULTS:
        # Prepare x-axis values
        epochs_range = range(1, epochs + 1)

        # Plot (1) Train and validation loss vs epochs
        plot_and_save(epochs_range, [train_losses, val_losses], ["Train Loss", "Validation Loss"],
                      "Train and Validation Loss", "Epochs", "Loss", "train_val_loss.png")

        # Plot (2) Dice score of each class vs epochs
        plot_and_save(epochs_range, dice_scores_per_class, [f"Class {i}" for i in range(6)],
                      "Dice Score per Class", "Epochs", "Dice Score", "dice_scores.png")
