import torch
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 144

TRAIN_IMG_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train"
TRAIN_MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_train"
VAL_IMG_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
VAL_MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"
TEST_IMG_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test"
TEST_MASK_DIR = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test"

N_LABELS = 6
LABEL_NUMBER_TO_NAME = {0: "Background", 1: "Body", 2: "Bone", 3: "Bladder", 4: "Rectum", 5: "Prostate"}

SEED = 47443349
def set_seed(seed=SEED):
    """
    For reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def dice_score(predictions, masks, smooth=1e-6):
    """
    Calculates dice Score for each class in multi-class segmentation.

    Parameters:
        predictions: Predicted logits of shape (B, C, H, W)
        masks: Ground truth masks of shape (B, 1, H, W) with label values in [0, C-1]

    Returns:
        Dice score for each class.
    """
    n_labels = predictions.shape[1]

    # One-hot encode masks
    masks = masks.squeeze(1).long() # (B, C, H, W) -> (B, H, W)
    masks_one_hot = F.one_hot(masks, num_classes=n_labels) # (B, H, W, C)
    masks_one_hot = masks_one_hot.permute(0, 3, 1, 2).float() # (B, C, H, W)

    # Apply softmax to get label probabilities
    label_probs = F.softmax(predictions, dim=1) # (B, C, H, W)

    # Calculate intersection and union
    intersection = (label_probs * masks_one_hot).sum(dim=(0, 2, 3))
    total = label_probs.sum(dim=(0, 2, 3)) + masks_one_hot.sum(dim=(0, 2, 3))

    # Dice score for each class
    dice_score = (2.0 * intersection + smooth) / (total + smooth)
    return dice_score
    

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, masks):
        """
        Dice Loss for multi-class segmentation.

        Parameters:
            predictions: Predicted logits of shape (B, C, H, W)
            masks: Ground truth masks of shape (B, 1, H, W) with label values in [0, C-1]

        Returns:
            Dice Loss (average over classes).
        """
        # Dice loss for each class
        dice_loss = 1 - dice_score(predictions, masks)

        return dice_loss.mean()
    