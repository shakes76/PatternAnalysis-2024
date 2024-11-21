import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K

import numpy as np

from matplotlib import pyplot as plt

#define training variables
BATCH_SIZE = 16
EPOCHS = 50
n_classes = 6
learning_rate = 0.0001
run = "drop0.2"

# Define the Dice similarity coefficient
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten ground truth tensor
    y_pred_f = K.flatten(y_pred)  # Flatten predicted tensor
    intersection = K.sum(y_true_f * y_pred_f)  # Intersection between true and predicted
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def calculate_dice_per_class(y_true, y_pred, num_classes):
    """
    Calculate Dice coefficient for each class.

    Args:
        y_true: Ground truth segmentation masks (one-hot encoded).
        y_pred: Predicted segmentation masks (one-hot encoded).
        num_classes: Number of classes.

    Returns:
        dice_scores: List of Dice coefficients for each class.
    """
    dice_scores = []

    for class_id in range(num_classes):
        true_class = (y_true == class_id).astype(np.float32)
        pred_class = (y_pred == class_id).astype(np.float32)

        dice = dice_coefficient(true_class, pred_class)
        dice_scores.append(dice.numpy())  # Convert to NumPy

    return dice_scores


# Function to save a validation image
def save_validation_image(image, mask, prediction, index):
    """Saves the original image, mask, and prediction."""
    
    # If the mask has more than one channel, convert it back to a single channel
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    # Similarly, for predictions
    if len(prediction.shape) == 4:
        prediction = prediction[:, :, :, 0]

    # Create a figure
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze())  
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask)  
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(prediction)  
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # Save the figure
    plt.savefig(f'validation_image_{run}_{index}.png')
    plt.close()


