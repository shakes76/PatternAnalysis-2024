import os
import time

import torch
import torch.nn.functional as F

from dataset import get_test_dataloader
from modules import UNet3D
from train import validate, Dice
from utils import (animate_segmentation, visualise_slices)
from config import MODEL_PATH, VISUALISE_RESULTS

if __name__ == '__main__':
    """
    Main entry point for testing the trained UNet3D model on a test dataset.
    Loads the model and computes Dice scores and loss on the test set.
    Visualizes the segmentation results.
    """
    # Initialize model
    device = torch.device("cuda:0")
    unet = UNet3D().to(device)
    test_loader = get_test_dataloader(batch_size=1)
    criterion = Dice()

    # Check if the model file exists
    if os.path.exists(MODEL_PATH):
        print("Model found, loading saved model...")
        unet.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        test_start_time = time.time()  # Start timer

        final_dice_scores, final_dice_loss = validate(unet, test_loader, criterion)

        test_end_time = time.time()  # End timer
        test_time = test_end_time - test_start_time  # Calculate elapsed time
        dice_coeff_str = ', '.join([f"{dc:.2f}" for dc in final_dice_scores])
        print(f"Final Dice Coefficients for each class: [{dice_coeff_str}]")
        print(f"Final Dice Loss: {final_dice_loss:.4f}")
        print(f"Total test time: {test_time:.2f} seconds")

        if VISUALISE_RESULTS:
            # Prepare data for visualisation
            first_batch = next(iter(test_loader))
            images, labels = first_batch["image"].to(device), first_batch["label"].to(device)
            with torch.no_grad():
                preds = unet(images)
            preds = torch.argmax(preds, dim=1)
            preds = F.one_hot(preds, num_classes=6).permute(0, 4, 1, 2, 3).float()

            images = images.cpu().numpy()
            preds = preds.cpu().numpy()
            labels = labels.cpu().numpy()

            # Create and save visualisations to running directory
            visualise_slices(images, labels, preds)
            animate_segmentation(images, labels, preds)

    else:
        print("No saved model found, cannot make predictions: try running train.py first")
