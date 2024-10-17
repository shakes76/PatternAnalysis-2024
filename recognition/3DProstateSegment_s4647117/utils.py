import torch
import torch.nn.functional as F

# Pre-computed class weights
CLASS_WEIGHTS = torch.tensor([0.2769, 0.4681, 4.9273, 28.8276, 117.5489, 161.1025])

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