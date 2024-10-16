"""
predict.py
----------
This script performs prediction using a trained U-Net model on test MRI slices.

Input:
    - Preprocessed MRI slices to predict on.
    - Trained model weights (specified by `model_path`).

Output:
    - Predicted segmentation masks visualized alongside original images.

Usage:
    Run this script to visualize predictions on a set of test MRI images.

Author: Han Yang
Date: 30/09/2024
"""
import torch
import numpy as np
from modules import UNet
from dataset import ProstateMRIDataset
import matplotlib.pyplot as plt

# Dice_score
def dice_score(pred, target, threshold=0.5, eps=1e-6):
    # Compute the Dice Similarity Coefficient (DSC).
    pred = (pred > threshold).float()  # Binarize predictions
    target = target.float()  # Ensure target is also float

    intersection = torch.sum(pred * target)
    dice = (2 * intersection + eps) / (torch.sum(pred) + torch.sum(target) + eps)
    return dice.item()  # Return a scalar


# Predicting images and evaluating model performance
def predict_and_evaluate(root_dir, model_path='unet_model.pth', threshold=0.5):
    # Load dataset
    dataset = ProstateMRIDataset(root_dir)
    model = UNet(n_channels=1, n_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # initialization
    total_dice_score = 0
    num_samples = len(dataset)

    # Not calculating gradient inference
    with torch.no_grad():
        for i in range(num_samples):
            image, ground_truth = dataset[i]
            output = model(image.unsqueeze(0))
            prediction = torch.sigmoid(output).squeeze(0).numpy()

            # Calculate Dice score
            dice = dice_score(torch.tensor(prediction), ground_truth)
            total_dice_score += dice

            # Optionally display some sample predictions
            if i < 5:
                plt.subplot(1, 2, 1)
                plt.imshow(image.squeeze(), cmap='gray')
                plt.title("Original Image")

                plt.subplot(1, 2, 2)
                plt.imshow(prediction, cmap='gray')
                plt.title("Prediction")

                plt.show()


# Program entrance
if __name__ == "__main__":
    root_dir = 'HipMRI_study_keras_slices_data/processed_nii_files'

    # Call and calculate Dice coefficient
    dice = predict_and_evaluate(root_dir)
    if dice >= 0.75:
        print(f"Model achieved the desired Dice score of 0.75 or above: {dice:.2f}")
    else:
        print(f"Model did not achieve the desired Dice score: {dice:.2f}")
