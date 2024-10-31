import time

import numpy as np
import torch

from dataset import get_dataloaders

from modules import UNet3D
from utils import plot_and_save, compute_class_weights
from config import MODEL_PATH

device = torch.device("cuda:0")


class Dice(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Initialize the Dice loss module.

        Args:
            smooth (float): Smoothing factor to avoid division by zero when calculating the Dice coefficient.
        """
        super(Dice, self).__init__()
        self.smooth = smooth

    def dice_scores_per_class(self, pred, target):
        """
        Compute the Dice coefficient for each class, excluding the background.

        Args:
            pred (torch.Tensor): Model predictions with shape (batch_size, classes, depth, height, width).
            target (torch.Tensor): Ground truth with shape (batch_size, classes, depth, height, width).

        Returns:
            torch.Tensor: Dice coefficients for each class excluding the background.
        """
        # Apply softmax to logits to get probabilities
        pred = torch.softmax(pred, dim=1)  # (B, C, H, W, D)

        # Exclude background by slicing from class 1 onward
        pred, target = pred[:, 1:], target[:, 1:]  # Skip the background class (C=0)

        # Define the axes for reduction (batch, depth, height, width)
        reduce_axis = [0] + list(range(2, len(pred.shape)))  # [0, 2, 3, 4]

        # Compute the intersection and union for each class
        intersection = torch.sum(pred * target, dim=reduce_axis)  # (num_classes - 1,)
        ground_o = torch.sum(target, dim=reduce_axis)  # (num_classes - 1,)
        pred_o = torch.sum(pred, dim=reduce_axis)  # (num_classes - 1,)

        # Compute the denominator for Dice coefficient
        denominator = ground_o + pred_o

        # Compute Dice coefficient for each class excluding background
        f = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return f

    def calculate_total_loss(self, class_dice_scores, target):
        class_weights = compute_class_weights(target)
        coeff = class_dice_scores * class_weights
        dice_loss = 1 - torch.sum(coeff) / torch.sum(class_weights)  # Weighted mean
        return dice_loss

    def forward(self, pred, target):
        """
        Compute the Dice loss, excluding background class.

        Args:
            pred (Tensor): Model outputs with shape (batch_size, num_classes, height, width, depth).
            target (Tensor): Ground truth with shape (batch_size, num_classes, height, width, depth).

        Returns:
            Tensor: Dice loss.
        """
        dice_scores = self.dice_scores_per_class(pred, target)
        return self.calculate_total_loss(dice_scores, target)


def train(model, dataloader, optimizer, crit, accumulation_steps=8):
    model.train()
    epoch_loss = 0
    torch.manual_seed(2809)  # reproducibility

    optimizer.zero_grad()  # Initialize gradients to zero
    batch_loss = 0  # To accumulate losses for logging

    for i, batch_data in enumerate(dataloader):
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        outputs = model(images)  # Forward pass
        loss = crit(outputs, labels)  # Compute loss

        # Scale the loss by the number of accumulation steps
        loss = loss / accumulation_steps
        loss.backward()  # Backward pass (accumulate gradients)

        batch_loss += loss.item()  # Accumulate loss for logging

        # Update model weights after accumulating gradients
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            optimizer.step()  # Perform optimization step
            optimizer.zero_grad()  # Reset gradients

        epoch_loss += loss.item() * accumulation_steps  # Accumulate total loss

    epoch_loss /= len(dataloader)  # Average the loss over all batches
    return epoch_loss


def validate(model, dataloader, crit):
    model.eval()  # Set model to evaluation mode
    dice_scores = []
    dice_losses = []

    with torch.no_grad():  # Disable gradient computation
        torch.manual_seed(2809)  # reproducibility
        for batch_data in dataloader:
            images, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            pred = model(images)  # Forward pass

            # dice scores per class
            new_dice_scores = crit.dice_scores_per_class(pred, labels)
            dice_scores.append(new_dice_scores.cpu().numpy())

            #dice loss
            dice_loss = crit.calculate_total_loss(new_dice_scores, labels) # Weighted mean
            dice_losses.append(dice_loss.cpu().numpy())

    dice_scores = np.mean(dice_scores, axis=0)
    dice_loss = np.mean(dice_losses)
    return dice_scores, dice_loss

if __name__ == '__main__':
    """
    Main function to run the training and validation processes.
    """

    # Set up datasets and DataLoaders
    train_loader, val_loader = get_dataloaders()

    # Initialize model
    unet = UNet3D(1, 6)
    unet = unet.to(device)

    epochs = 20
    criterion = Dice()
    optimizer = torch.optim.Adam(unet.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_metric = float(100.)
    best_state = unet.state_dict()

    train_losses, val_losses = [], []
    dice_scores_per_class = [[] for _ in range(6)]
    lrs, weight_decays = [], []

    train_start_time = time.time()

    # Training and evaluation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train(unet, train_loader, optimizer, criterion)
        train_losses.append(train_loss)

        print(f"Train Loss: {train_loss:.4f}")

        dice_scores, val_loss = validate(unet, val_loader, criterion)
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

