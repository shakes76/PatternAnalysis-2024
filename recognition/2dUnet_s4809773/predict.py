"""
Author: Roman Kull
Description: 
    This file takes in a trained model and compares its performance to true segmentations from the validation set
    It outputs the model's dice coefficient for every segmentation class
    Additionally, outputs 3 different sets of images, doing a side by side comparison of the MRI slice, the true segmentation, and the predicted segmentation
"""

import torch
import torch.nn.functional as F
from dataset import create_dataloaders  # Import dataset and loader
from modules import UNet  # Import the UNet model
import numpy as np
import matplotlib.pyplot as plt
import random

# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_classes = 6  # As there are 6 segmentation classes

# Path to the trained model file
model_path = "unet_model.pth"


# Paths to the validate dataset data
validate_images_folder = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_validate"
validate_masks_folder = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_validate"

# Load the dataset
validation_loader = create_dataloaders(validate_images_folder, validate_masks_folder, batch_size, normImage=True)

# Initialize the model, load it with the pre-trained model, and set to evaluation mode
net = UNet(num_classes=num_classes).to(device)  # Move the model to the device
net.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
net.eval()

# Dice Loss Function for individual class loss calculations
def dice_loss_per_class(pred, target, num_classes, smooth=1):
    # Apply softmax to get probabilities for each class
    pred = F.softmax(pred, dim=1)
    
    # Ensure target is properly shaped as [batch_size, H, W]
    target = target.squeeze(dim=1) if target.dim() == 4 else target
    
    # Convert target to one-hot encoding with the correct shape
    target_flat = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).contiguous().view(pred.shape[0], num_classes, -1)  # [batch_size, num_classes, H*W]
    
    # Flatten pred tensor to match the target
    pred_flat = pred.view(pred.shape[0], num_classes, -1)  # [batch_size, num_classes, H*W]
    
    # Calculate intersection and Dice score per class
    intersection = (pred_flat * target_flat).sum(dim=2)  # Sum over H*W (spatial dimensions)
    pred_sum = pred_flat.sum(dim=2)
    target_sum = target_flat.sum(dim=2)
    
    # Calculate Dice score for each class (batch_size, num_classes)
    dice_score = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    
    # Calculate Dice loss per class (1 - Dice score)
    per_class_loss = 1 - dice_score.mean(dim=0)  # Average across the batch for each class

    return per_class_loss

# Function to visualize the original, true, and predicted segmentation
def visualize_predictions(model, data_loader, device, num_classes):
    dataset_size = len(data_loader.dataset)
    indices = random.sample(range(dataset_size), 3)  # Get three random indices from the dataset
    samples = [data_loader.dataset[i] for i in indices]

    for idx, (image, true_mask) in enumerate(samples):
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        true_mask = true_mask.squeeze().cpu().numpy()

        # Predict the segmentation
        with torch.no_grad():
            pred_mask = model(image)
            pred_mask = torch.argmax(pred_mask, dim=1).squeeze().cpu().numpy()

        # Create a 1x3 plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
        axes[0].set_title("Original Image")
        axes[1].imshow(true_mask, cmap='gray')
        axes[1].set_title("True Segmentation")
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title("Predicted Segmentation")

        for ax in axes:
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'validation_{idx + 1}.png')  # Save each plot with a unique name
        plt.close()

# Function to evaluate the model using Dice loss per class
def evaluate_model(model, data_loader, device, num_classes):
    total_dice_per_class = np.zeros(num_classes)
    batch_count = 0

    with torch.no_grad():
        for images, true_masks in data_loader:
            images = images.to(device)
            true_masks = true_masks.to(device)

            # Forward pass to get predictions
            pred_masks = model(images)
            
            # Calculate Dice loss per class
            per_class_loss = dice_loss_per_class(pred_masks, true_masks, num_classes)
            
            # Convert Dice loss to Dice score for each class
            per_class_dice = 1 - per_class_loss.cpu().numpy()
            
            # Accumulate the Dice scores for averaging
            total_dice_per_class += per_class_dice
            batch_count += 1

    # Average the Dice scores across all batches
    average_dice_per_class = total_dice_per_class / batch_count
    print(f"Average Dice Score per Class:")
    for i, dice_score in enumerate(average_dice_per_class):
        print(f"Class {i}: {dice_score:.4f}")

    print(f"Overall Average Dice Score: {average_dice_per_class.mean():.4f}")

# Main function to run the predictions and evaluations
if __name__ == "__main__":
    # Evaluate model and print average class accuracy
    evaluate_model(net, validation_loader, device, num_classes)

    # Visualize three random samples
    visualize_predictions(net, validation_loader, device, num_classes)
