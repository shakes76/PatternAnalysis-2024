import torch
import torch.nn as nn
import torch.nn.functional as F

# class pixel counts for weighted loss calculations
class_pixel_counts = {
    0: 1068883043,
    1: 627980239,
    2: 59685345,
    3: 10172936,
    4: 2551801,
    5: 1771500,
}
total_pixels = sum(class_pixel_counts.values())

# Set the device for training (CUDA if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")

NUM_CLASSES = 6  # Number of classes in the dataset

# Compute weights for each class to handle class imbalance
class_weights = torch.tensor(
    [total_pixels / (NUM_CLASSES * class_pixel_counts[c]) for c in range(NUM_CLASSES)],
    dtype=torch.float32,
    device=DEVICE
)

def weighted_cross_entropy_loss():
    """
    Returns a cross-entropy loss function weighted by class frequencies.
    :return: Instance of nn.CrossEntropyLoss with class weights
    """
    return nn.CrossEntropyLoss(weight=class_weights)

class DiceLoss(nn.Module):
    """
    Implements the Dice Loss for evaluating the overlap between predicted and ground-truth segmentation.
    """
    def forward(self, inputs, targets):
        smooth = 1e-6
        inputs_softmax = torch.softmax(inputs, dim=1)  # Convert logits to probabilities
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).permute(0, 4, 1, 2, 3).float()

        # Compute intersection and union
        intersection = torch.sum(inputs_softmax * targets_one_hot, dim=(2, 3, 4))
        union = torch.sum(inputs_softmax + targets_one_hot, dim=(2, 3, 4))

        # Calculate Dice score
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - torch.mean(dice)

class CombinedLoss(nn.Module):
    """
    Combines Weighted Cross-Entropy Loss and Dice Loss into a single composite loss function.
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = weighted_cross_entropy_loss()  # Initialize weighted cross-entropy loss
        self.dice_loss = DiceLoss()  # Initialize Dice loss

    def forward(self, inputs, targets):
        ce = self.ce_loss(inputs, targets)  # Compute cross-entropy loss
        dice = self.dice_loss(inputs, targets)  # Compute Dice loss
        log_dice = torch.log(dice + 1e-6)  # Log-transform of Dice loss to stabilize gradients

        # Compute the combined loss
        return self.ce_weight * ce + self.dice_weight * log_dice
