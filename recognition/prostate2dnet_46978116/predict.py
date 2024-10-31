# predict.py

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import ProstateDataset
from modules import UNet
from train import CombinedLoss, validate_fn, save_img, plot_metrics  
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration Parameters
batch_size = 64
n_workers = 2
pin = True
device = "cuda" if torch.cuda.is_available() else "cpu"
img_height = 256
img_width = 128
num_classes = 6
best_model_path = "savedmodels/bestDice.pth"  # Path to the saved best model
test_image_dir = 'keras_slices_data/keras_slices_test'
test_mask_dir = 'keras_slices_data/keras_slices_seg_test'
output_folder = 'dice_test_predictions'

def main():
    # Define transformation pipeline for testing (no augmentations, only resizing and normalization)
    test_transform = A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])

    # Instantiate the test dataset and DataLoader
    test_dataset = ProstateDataset(
        image_path=test_image_dir,
        mask_path=test_mask_dir,
        norm_image=True,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin
    )

    # Initialize the model and load the best saved weights
    model = UNet().to(device=device)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model weights from '{best_model_path}'.")
    else:
        print(f"Best model not found at '{best_model_path}'. Exiting.")
        return

    # Define the loss function (same as used during training)
    ce_weights = torch.tensor([1, 1, 1, 2, 10, 4], dtype=torch.float).to(device)  # Ensure these match training
    loss_fn = CombinedLoss(ce_weight=ce_weights)

    # Evaluate the model on the test set and save dice scores
    print("Evaluating the model on the test set...")
    test_loss, test_dice = validate_fn(test_loader, model, loss_fn)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice Coefficients per class: {test_dice}")

    # Save prediction images for qualitative assessment
    print(f"Saving prediction images to '{output_folder}'...")
    save_img(test_loader, model, folder=output_folder, device=device, num_classes=num_classes, epoch='test')
    save_img(test_loader, model, folder=output_folder + 'small', device=device, num_classes=num_classes, epoch='test',max_images_per_class=2)
    # Compute overall Dice score (average across classes)
    average_dice = np.mean(list(test_dice.values()))
    print(f"Average Dice Score across all classes: {average_dice:.4f}")

    # save dice scores to file
    with open(os.path.join(output_folder, 'test_dice_scores.txt'), 'w') as f:
        for cls, score in test_dice.items():
            f.write(f"Class {cls}: {score:.4f}\n")
        f.write(f"Average Dice Score: {average_dice:.4f}\n")
    print(f"Dice scores saved to '{output_folder}/test_dice_scores.txt'.")

if __name__ == '__main__':
    main()
