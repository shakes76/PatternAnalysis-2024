import torch
from utils import one_hot_mask


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
            pred (torch.Tensor): Model predictions with shape (batch_size, num_classes, depth, height, width).
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
