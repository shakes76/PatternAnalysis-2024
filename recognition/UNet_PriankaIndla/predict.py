import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules import UNet  # Import UNet class from modules.py
from dataset import ProstateMRIDataset  # Import ProstateMRIDataset class from dataset.py

# Directories for test images and segmentations
TEST_IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_test'
TEST_SEGMENTATION_DIR = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_seg_test'

# Model parameters
INPUT_IMAGE_HEIGHT = 256 
INPUT_IMAGE_WIDTH = 128   
BATCH_SIZE = 1
TARGET_SHAPE = (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)

# Load test data
print("Loading test image paths...")
test_image_paths = [os.path.join(TEST_IMAGE_DIR, f) for f in os.listdir(TEST_IMAGE_DIR)]
test_seg_image_paths = [os.path.join(TEST_SEGMENTATION_DIR, f) for f in os.listdir(TEST_SEGMENTATION_DIR)]
print(f"Total test images: {len(test_image_paths)}, Total test segmentations: {len(test_seg_image_paths)}")

# Create test dataset and dataloader
test_dataset = ProstateMRIDataset(test_seg_image_paths, test_image_paths, normImage=True, target_shape=TARGET_SHAPE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check GPU is working
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialise model with the correct input and output channels for 1-channel images
model = UNet(in_channels=1, out_channels=1, retainDim=True, outSize=(INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)).to(device)

# Load best model
model.load_state_dict(torch.load('best_model.pth', map_location=device))  

# Set model to evaluation mode
model.eval()  
print("Model loaded and set to evaluation mode.")

def visualise_prediction(image, prediction, ground_truth=None, save_path=None):
    """
    Visualise the original image, predicted segmentation, and ground truth segmentation
    
    Args:
        image (numpy array): Original MRI image.
        prediction (numpy array): Model's predicted segmentation.
        ground_truth (numpy array, optional): Actual segmentation mask.
        save_path (str, optional): Path to save the visualisation. Defaults to None.
    """
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Prediction
    plt.subplot(1, 3, 2)
    plt.imshow(prediction.squeeze(), cmap='gray')
    plt.title("Predicted Segmentation")
    plt.axis('off')

    # Ground truth 
    if ground_truth is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(ground_truth.squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis('off')

    # Save visualisation to path
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction saved as {save_path}")

    plt.show()

# Run predictions on the test dataset
with torch.no_grad():
    for idx, (images, segs) in enumerate(test_loader):
        images = images.to(device)

        # Generate prediction
        outputs = model(images)
        predicted_segmentation = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        predicted_segmentation = (predicted_segmentation > 0.5).float()  # Threshold the prediction

        # Convert to numpy for visualisation
        image_np = images.cpu().numpy()
        predicted_segmentation_np = predicted_segmentation.cpu().numpy()
        ground_truth_np = segs.cpu().numpy()

        # Visualise and save prediction
        visualise_prediction(image_np, predicted_segmentation_np, ground_truth_np,
                             save_path=f"prediction_{idx}.png")

        if idx == 4:  # Display and save predictions for the first 5 images
            break


