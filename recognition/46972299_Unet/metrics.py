"""
Contains the code for calculating losses and metrics for the Unet
"""

# will use https://smp.readthedocs.io/en/latest/index.html for Dice loss
from segmentation_models_pytorch.losses import DiceLoss

SMOOTH_FACTOR = 1e-7

def get_loss_function() -> DiceLoss:
    return DiceLoss('multiclass', from_logits=False, smooth=SMOOTH_FACTOR)

