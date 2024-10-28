import time

import numpy as np
import torch
from utils import one_hot_mask
from dataset import get_dataloaders
from modules import UNet3D
from torch.utils.tensorboard import SummaryWriter

MODEL_PATH = "best_unet.pth"
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
        input = torch.softmax(pred, dim=1)  # (B, C, D, H, W)

        # Convert target to one-hot encoding along the class dimension
        target = one_hot_mask(target)  # (B, C, D, H, W)

        # Define the axes for reduction (batch, depth, height, width)
        reduce_axis = [0] + list(range(2, len(input.shape)))  # [0, 2, 3, 4]

        # Compute the intersection and union for each class
        intersection = torch.sum(input * target, dim=reduce_axis)  # (num_classes,)
        ground_o = torch.sum(target, dim=reduce_axis)  # (num_classes,)
        pred_o = torch.sum(input, dim=reduce_axis)  # (num_classes,)

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


def train(model, dataloader, optimizer, crit):
    model.train()
    epoch_loss = 0
    torch.manual_seed(2809)  # reproducibility
    for batch_data in dataloader:
        images, masks = batch_data["img"].to(device), batch_data["mask"].to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = crit(outputs, masks)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        epoch_loss += loss.item()  # Accumulate loss
    return epoch_loss


def validate(model, dataloader, crit):
    model.eval()  # Set model to evaluation mode
    dice_scores = []

    with torch.no_grad():  # Disable gradient computation
        torch.manual_seed(2809)  # reproducibility
        for batch_data in dataloader:
            imgs, masks = batch_data["img"].to(device), batch_data["mask"].to(device)
            pred = model(imgs)  # Forward pass
            new_dice_score = crit.dice(pred, masks)
            dice_scores.append(new_dice_score.cpu().numpy())

    dice_scores = np.mean(dice_scores, axis=0)
    return dice_scores


# TODO: Test method with visualisation


if __name__ == '__main__':
    """
    Main function to run the training and validation processes.
    """
    writer = SummaryWriter(log_dir="./runs/unet_training")  # Specify the log directory

    # Set up datasets and DataLoaders
    batch_size = 8
    train_loader, val_loader, test_loader = get_dataloaders(train_batch=batch_size, val_batch=batch_size)

    # Initialize model
    unet = UNet3D()
    unet = unet.to(device)

    epochs = 15
    criterion = Dice()
    optimizer = torch.optim.Adam(unet.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_metric = float(0.)
    best_state = unet.state_dict()

    train_start_time = time.time()

    # Training and evaluation loop
    for epoch in range(epochs):
        # # Training
        train_loss = train(unet, train_loader, optimizer, criterion)
        train_loss_avg = train_loss / len(train_loader)
        print(f"Train Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss_avg:.4f}")

        # Log training loss to TensorBoard
        writer.add_scalar("Loss/Train", train_loss_avg, epoch)

        # Log learning rate for each param group in the optimizer
        for param_group_idx, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f"LR/Group_{param_group_idx}", param_group['lr'], epoch)
            writer.add_scalar(f"Weight_Decay/Group_{param_group_idx}", param_group['weight_decay'], epoch)
        scheduler.step()

        # # Validation
        dice_scores = validate(unet, val_loader, criterion)
        dice_coeff_str = ', '.join([f"{dc:.2f}" for dc in dice_scores])
        print(f"Test Epoch {epoch + 1}/{epochs}, Dice Coefficients: [{dice_coeff_str}]")

        # Log validation loss to TensorBoard
        validation_loss = 1. - float(np.mean(dice_scores))  # Average validation loss
        writer.add_scalar("Loss/Validation", validation_loss, epoch)

        # Log Dice score for each class to TensorBoard
        for class_idx, dice_scores in enumerate(dice_scores):
            writer.add_scalar(f"Dice_Score/Class_{class_idx}", dice_scores, epoch)

        # if this is the best performance so far, save this state
        if validation_loss < best_metric:
            best_metric = validation_loss
            best_state = unet.state_dict()
            torch.save(best_state, MODEL_PATH)

    train_time = time.time() - train_start_time  # Calculate elapsed time
    print(f"Total training time: {train_time:.2f} seconds")

    # Load the best model state (if not loaded already)
    unet.load_state_dict(best_state)

    # test the model on separate test dataset
    test_start_time = time.time()  # Start timer
    final_dice_score = validate(unet, test_loader, criterion)

    test_time = time.time() - test_start_time  # Calculate elapsed time
    dice_coeff_str = ', '.join([f"{dc:.2f}" for dc in final_dice_score])
    print(f"Final Dice Coefficients for each class: [{dice_coeff_str}]")
    print(f"Total test time: {test_time:.2f} seconds")
    writer.close()
