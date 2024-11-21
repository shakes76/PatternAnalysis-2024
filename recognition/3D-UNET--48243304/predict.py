import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset  # Import dataset for inference
from modules import UNet3D  # Import the 3D U-Net model
import os
os.environ['MPLCONFIGDIR'] = '/home/Student/s4824330/3D-UNET/predictions'
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNet3D(in_channels=1, out_channels=1, init_features=32).to(device)
model.load_state_dict(torch.load('unet3d_model.pth', weights_only=False))  # Load saved model weights
model.eval()  # Set the model to evaluation mode

# Define the test dataset paths
test_data_dir = r"/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
label_data_dir = r"/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

# Load test data (no labels needed for prediction)
test_dataset = Prostate3DDataset(data_dir=test_data_dir, label_dir=label_data_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Function to calculate Dice Coefficient
def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=(1, 2, 3))  # Summing over spatial dimensions
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()  # Return average Dice coefficient for batch

# Function to plot and save grid of predictions
def plot_predictions_grid(image_np, label_np, output_np, slices_to_display, output_dir, i):
    fig, axes = plt.subplots(3, len(slices_to_display), figsize=(15, 10))
    
    for j, slice_idx in enumerate(slices_to_display):
        # Original image
        axes[0, j].imshow(image_np[slice_idx], cmap="gray")
        axes[0, j].set_title(f"Image Slice {slice_idx}")
        
        # Ground truth label with color mapping
        axes[1, j].imshow(image_np[slice_idx], cmap="gray")  # Background grayscale image
        axes[1, j].imshow(label_np[slice_idx], cmap="jet", alpha=0.5)  # Overlay label with color mapping
        axes[1, j].set_title(f"Label Slice {slice_idx}")
        
        # Predicted output with color mapping
        axes[2, j].imshow(image_np[slice_idx], cmap="gray")  # Background grayscale image
        axes[2, j].imshow(output_np[slice_idx], cmap="jet", alpha=0.5)  # Overlay prediction with color mapping
        axes[2, j].set_title(f"Prediction Slice {slice_idx}")
        
    plt.tight_layout()
    plt_path = os.path.join(output_dir, f'prediction_grid_{i}.png')
    plt.savefig(plt_path)
    print(f"Saved prediction grid at {plt_path}")
    plt.close(fig)

# Define prediction function
def predict_and_evaluate(model, test_loader, output_dir, max_predictions=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.eval()  # Set the model to evaluation mode
    total_dice = 0
    num_samples = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)

            # Forward pass
            output = model(image)
            output = (output > 0.5).float()  # Threshold the segmentation output

            # Calculate Dice Coefficient for the current sample
            dice_score = dice_coefficient(output, label)
            total_dice += dice_score
            num_samples += 1
            print(f'Sample {i+1} Dice Coefficient: {dice_score:.4f}')

            # Only save the first max_predictions samples
            if i < max_predictions:
                output_np = output.cpu().numpy().squeeze(0).squeeze(0)  # Remove batch and channel dims
                image_np = image.cpu().numpy().squeeze(0).squeeze(0)
                label_np = label.cpu().numpy().squeeze(0).squeeze(0)
                
                # Define slices to display (top, middle, bottom)
                slices_to_display = [0, output_np.shape[0] // 2, output_np.shape[0] - 1]
                plot_predictions_grid(image_np, label_np, output_np, slices_to_display, output_dir, i)

            # Stop saving after reaching the max_predictions limit
            if i >= max_predictions - 1:
                break

    # Print average Dice coefficient over all test samples
    avg_dice = total_dice / num_samples
    print(f'Average Dice Coefficient: {avg_dice:.4f}')

# Perform predictions and evaluate
if __name__ == '__main__':
    output_dir = '/home/Student/s4824330/3D-UNET/predictions'  # Directory to save predictions
    max_predictions = 1  # Set the number of predictions to save (set to 1 for demonstration)
    predict_and_evaluate(model, test_loader, output_dir, max_predictions)
