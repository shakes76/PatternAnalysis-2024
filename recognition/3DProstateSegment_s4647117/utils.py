"""
This module provides utility functions for calculating 
weighted Dice loss and evaluating per-class Dice scores.

Functions:
- **per_class_dice_loss**: Computes the Dice loss for each class independently.
- **weighted_dice_loss**: Computes the weighted sum of per-class Dice losses using class-specific weights.
- **per_class_dice_components**: Computes the intersection and union for Dice score per class without one-hot encoding.

@author: Joseph Savage
"""
import torch
import torch.nn.functional as F

# Hyperparameter to prioritise classes that are less common to ensure they are not neglected
CLASS_WEIGHTS = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
CLASS_WEIGHTS /= CLASS_WEIGHTS.sum()

def per_class_dice_loss(pred, target, num_classes=6, epsilon=1e-6):
    """
    Computes Dice loss for each class separately.

    Args:
        pred (torch.Tensor): Predicted logits with shape (batch_size, num_classes, D, H, W).
        target (torch.Tensor): Ground truth labels with shape (batch_size, D, H, W).
        num_classes (int): Number of classes.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Dice loss per class with shape (num_classes,).
    """
    pred_probs = F.softmax(pred, dim=1)  # Convert logits to probabilities
    target_one_hot = F.one_hot(target, num_classes=num_classes)  # One-hot encode targets
    target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # Reshape to (batch_size, num_classes, D, H, W)

    # Flatten spatial dimensions
    pred_flat = pred_probs.view(pred_probs.size(0), num_classes, -1)
    target_flat = target_one_hot.view(target_one_hot.size(0), num_classes, -1)

    intersection = (pred_flat * target_flat).sum(-1)
    union = pred_flat.sum(-1) + target_flat.sum(-1)

    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    dice_loss = 1 - dice_score  # Shape: (batch_size, num_classes)

    # Average over the batch
    dice_loss = dice_loss.mean(dim=0)  # Shape: (num_classes,)
    return dice_loss  # Returns loss per class


def per_class_dice_components(pred, target, num_classes=6):
    """
    Computes the intersection and union for Dice score per class without one-hot encoding.

    Args:
        pred (torch.Tensor): Predicted logits with shape (batch_size, num_classes, D, H, W).
        target (torch.Tensor): Ground truth labels with shape (batch_size, D, H, W).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Intersection and union per class.
    """

    # Convert logits to predicted labels
    pred_labels = torch.argmax(pred, dim=1)  # Shape: (batch_size, D, H, W)
    
    # Flatten tensors to 1D
    pred_flat = pred_labels.view(-1)  # Shape: (N,)
    target_flat = target.view(-1)     # Shape: (N,)

    # Initialize tensors to hold counts
    intersection = torch.zeros(num_classes, device=pred.device)
    union = torch.zeros(num_classes, device=pred.device)

    # Compute intersection and union without explicit loops
    for cls in range(num_classes):
        # Create boolean masks for the current class
        pred_mask = (pred_flat == cls)
        target_mask = (target_flat == cls)

        # Compute intersection and union for the current class
        intersection[cls] = (pred_mask & target_mask).sum().float()
        union[cls] = pred_mask.sum().float() + target_mask.sum().float()

    return intersection, union


def weighted_dice_loss(pred, target, num_classes=6, class_weights=CLASS_WEIGHTS, epsilon=1e-6):
    """
    Computes weighted sum of per-class Dice losses.

    Args:
        pred (torch.Tensor): Predicted logits with shape (batch_size, num_classes, D, H, W).
        target (torch.Tensor): Ground truth labels with shape (batch_size, D, H, W).
        num_classes (int): Number of classes.
        class_weights (torch.Tensor or None): Weights for each class. Shape: (num_classes,).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: Weighted Dice loss (scalar).
    """
    dice_loss = per_class_dice_loss(pred, target, num_classes, epsilon)  # Shape: (num_classes,)

    if class_weights is not None:
        # Ensure class_weights is a tensor on the same device as dice_loss
        class_weights = class_weights.to(dice_loss.device)
        weighted_loss = dice_loss * class_weights
    else:
        weighted_loss = dice_loss  # Equal weights if none provided

    return weighted_loss.sum()  # Scalar loss