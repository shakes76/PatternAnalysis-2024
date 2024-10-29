import os
import time

import torch

from dataset import get_dataloaders, get_test_dataloader
from modules import UNet3D
from train import validate
from config import MODEL_PATH

if __name__ == '__main__':
    # Initialize model
    unet = UNet3D()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    unet = unet.to(device)

    batch_size = 8
    test_loader = get_test_dataloader(batch_size)

    # Check if the model file exists
    if os.path.exists(MODEL_PATH):
        print("Model found, loading saved model...")
        unet.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

        # TODO: Add validate function from train when built

        test_start_time = time.time()  # Start timer
        final_dice_score = validate()

        test_end_time = time.time()  # End timer
        test_time = test_end_time - test_start_time  # Calculate elapsed time
        dice_coeff_str = ', '.join([f"{dc:.2f}" for dc in final_dice_score])
        print(f"Final Dice Coefficients for each class: [{dice_coeff_str}]")
        print(f"Total test time: {test_time:.2f} seconds")

    else:
        print("No saved model found, cannot make predictions: try running train.py first")

    # TODO: figure out something for visualisation