"""
Prediction Script for 3D U-Net Model on Medical Image Segmentation

This script loads the trained 3D U-Net model and performs inference on a subset 
of test images. It selects three random images from the test set, 
predicts the segmentation, and saves the results as NIfTI files for visualisation. 
Additionally, it calculates and prints per-class Dice scores for each prediction, 
providing insights into the model's segmentation performance on each class.

@author Joseph Savage  
"""
import os
import random
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader

from modules import UNet3D
from dataset import NiftiDataset
from utils import per_class_dice_components

# Paths to test data
images_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
labels_path = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
model_path = 'unet_model.pth'

# Load the test dataset
test_dataset = NiftiDataset(
    image_filenames=sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.nii.gz')]),
    label_filenames=sorted([os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.nii.gz')]),
    dtype=np.float32,
    transform=None
)

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D().to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

# Select 3 random samples from the test set
sample_indices = random.sample(range(len(test_dataset)), 3)
sample_loader = DataLoader([test_dataset[i] for i in sample_indices], batch_size=1, shuffle=False)

# Function to save predictions and calculate Dice score
def predict_and_evaluate():
    for i, (image, label) in enumerate(sample_loader):
        image = image.to(device)
        label = label.squeeze(1).to(device)

        # Predict segmentation
        with torch.no_grad():
            output = model(image)
            pred_class = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Save the prediction as a NIfTI file
        pred_img = nib.Nifti1Image(pred_class.astype(np.int16), np.eye(4))
        nib.save(pred_img, f"predicted_image_{i+1}.nii.gz")
        print(f"Saved prediction for sample {i+1}.")

        # Calculate per-class Dice loss
        intersection, union = per_class_dice_components(output, label, num_classes=6)
        epsilon = 1e-6
        dice_scores = (2 * intersection + epsilon) / (union + epsilon)

        print(f"Per-class Dice scores for sample {i+1}: {dice_scores.cpu().numpy()}")

if __name__ == "__main__":
    predict_and_evaluate()
