import time

import numpy as np
import torch

from dataset import get_dataloaders
from modules import UNet3D
from utils import plot_and_save, MODEL_PATH

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
        # target = one_hot(target.squeeze(1).long(), num_classes=6).permute(0, 4, 1, 2, 3).float()
        pred = torch.softmax(pred, dim=1)  # (B, C, H, W, D)

        # Exclude background by slicing from class 1 onward
        pred, target = pred[:, 1:], target[:, 1:]  # Skip the background class (C=0)

        # Define the axes for reduction (batch, depth, height, width)
        reduce_axis = [0] + list(range(2, len(pred.shape)))  # [0, 2, 3, 4]

        # Compute the intersection and union for each class
        intersection = torch.sum(pred * target, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(pred, dim=reduce_axis)

        # Compute Dice coefficient for each class excluding background
        f = (2.0 * intersection + self.smooth) / (ground_o + pred_o + self.smooth)

        return f

    def calculate_loss(self, class_dice_scores):
        return 1 - torch.mean(class_dice_scores)

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
        return self.calculate_loss(dice_scores)


def train(model, dataloader, optimizer, crit, accumulation_steps=8):
    model.train()
    epoch_loss = 0
    batch_dice_losses = []
    torch.manual_seed(2809)  # reproducibility

    for i, batch_data in enumerate(dataloader):
        images, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        # Forward pass
        outputs = model(images)
        loss = crit(outputs, labels)  # Compute loss
        batch_dice_losses.append(loss.item())

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
    return epoch_loss, batch_dice_losses


def validate(model, dataloader, crit):
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
    # Set up datasets and DataLoaders
    train_loader, val_loader = get_dataloaders()

    # Initialize model
    unet = UNet3D()
    unet = unet.to(device)

    epochs = 18
    lr = 0.001
    accumulation_steps = 8
    batch_size = 2
    criterion = Dice()
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    # tracking best model
    best_metric = float(100.)
    best_state = unet.state_dict()

    # logging info
    train_losses, val_losses = [], []
    dice_scores_per_class = [[] for _ in range(5)]
    batch_dice_loss_history = []

    train_start_time = time.time()
    print(f"training parameters: {epochs} epochs, {batch_size} batch size, {lr} learning rate, "
          f"{accumulation_steps} gradient accumulation steps")
    # Training and evaluation loop
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, batch_dice_losses = train(unet, train_loader, optimizer, criterion, accumulation_steps)
        train_losses.append(train_loss)
        batch_dice_loss_history.extend(batch_dice_losses)

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

    iterations = range(1, len(batch_dice_loss_history) + 1)  # x-axis for the iterations
    plot_and_save(iterations, batch_dice_loss_history, "Dice Loss","Train Dice Loss Over Iterations",
                  "Iterations", "Dice Loss", "train_dice_loss_iterations.png")
