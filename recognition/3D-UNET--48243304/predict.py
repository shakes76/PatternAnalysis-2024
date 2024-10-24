import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader
from dataset import Prostate3DDataset  # Import dataset for inference
from modules import UNet3D  # Import the 3D U-Net model
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = UNet3D(in_channels=1, out_channels=1, init_features=16).to(device)
model.load_state_dict(torch.load('unet3d_model.pth'))  # Load saved model weights
model.eval()  # Set the model to evaluation mode

# Define the test dataset paths
test_data_dir = r"/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
label_data_dir = r"/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

# Load test data (no labels needed for prediction)
test_dataset = Prostate3DDataset(data_dir=test_data_dir, label_dir=label_data_dir)

# Create PyTorch DataLoader for test data
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Function to calculate Dice Coefficient
def dice_coefficient(pred, target, smooth=1e-5):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=(1, 2, 3))  # Summing over spatial dimensions
    dice = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
    return dice.mean().item()  # Return average Dice coefficient for batch

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

            # Save the output as Nifti file (only save the first max_predictions samples)
            if i < max_predictions:
                output_np = output.cpu().numpy().squeeze(0).squeeze(0)  # Remove batch and channel dims
                affine = np.eye(4)  # Create a simple identity affine matrix for Nifti
                output_nifti = nib.Nifti1Image(output_np, affine)
                output_path = os.path.join(output_dir, f'prediction_{i}.nii.gz')
                nib.save(output_nifti, output_path)
                print(f'Saved: {output_path}')
            
            # Stop saving after reaching the max_predictions limit
            if i >= max_predictions - 1:
                break

    # Print average Dice coefficient over all test samples
    avg_dice = total_dice / num_samples
    print(f'Average Dice Coefficient: {avg_dice:.4f}')

# Perform predictions and evaluate
if __name__ == '__main__':
    output_dir = './predictions'  # Directory to save predictions
    max_predictions = 1  # Set the number of predictions to save (set to 1 for demonstration)
    predict_and_evaluate(model, test_loader, output_dir, max_predictions)
