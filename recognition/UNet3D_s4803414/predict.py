import os
import torch
import matplotlib.pyplot as plt
from modules import UNet3D
from torch.utils.data import DataLoader
from torch.amp import autocast
import nibabel as nib

# Directories and parameters
MODEL_PATH = '/home/Student/s4803414/miniconda3/model/new_model.pth'
IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs'
MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only'

VISUALS_DIR = '/home/Student/s4803414/miniconda3/visuals'
BATCH_SIZE = 2
NUM_CLASSES = 6

# Ensure the visuals directory exists
os.makedirs(VISUALS_DIR, exist_ok=True)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

def save_visualization(image_slice, mask_slice, pred_slice, slice_idx, save_path):
    # Create the figure for each slice
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image slice
    axes[0].imshow(image_slice, cmap='grey')
    axes[0].set_title(f'Original Image Slice {slice_idx}')
    axes[0].axis('off')

    # Plot mask slice
    axes[1].imshow(mask_slice, cmap='viridis')
    axes[1].set_title(f'Mask Slice {slice_idx}')
    axes[1].axis('off')

    # Plot predicted slice
    axes[2].imshow(pred_slice, cmap='viridis')
    axes[2].set_title(f'Prediction Slice {slice_idx}')
    axes[2].axis('off')

    # Save the figure
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'visualization_slice_{slice_idx}.png'))
    plt.close()

# Get all NIfTI file paths
image_filenames = sorted(os.listdir(IMAGE_DIR))
mask_filenames = sorted(os.listdir(MASK_DIR))

# Prediction loop and saving visualizations
with torch.no_grad():
    for batch_idx in range(0, len(image_filenames), BATCH_SIZE):
        # Load the batch of images and masks directly from NIfTI files
        images = []
        masks = []

        for i in range(batch_idx, min(batch_idx + BATCH_SIZE, len(image_filenames))):
            img_path = os.path.join(IMAGE_DIR, image_filenames[i])
            mask_path = os.path.join(MASK_DIR, mask_filenames[i])

            img_mask = nib.load(img_path)
            img_data = img_mask.get_fdata()
            images.append(img_data)

            mask_mask = nib.load(mask_path)
            mask_data = mask_mask.get_fdata()
            masks.append(mask_data)

        # Convert images and masks to tensors
        images_tensor = torch.tensor(images).unsqueeze(1).float().to(device)  # Add channel dimension, cast to float32
        masks_tensor = torch.tensor(masks).unsqueeze(1).float().to(device)  # Add channel dimension, cast to float32

        with autocast(device_type='cuda'):
            outputs = model(images_tensor)
            predictions = torch.argmax(outputs, dim=1)  # Get the predicted class for each pixel

        # Process all images in the batch
        for i in range(images_tensor.size(0)):
            image = images_tensor[i].cpu().numpy().squeeze()  # Convert to numpy
            mask = masks_tensor[i].cpu().numpy().squeeze()    # Convert to numpy
            pred = predictions[i].cpu().numpy()    # Convert to numpy

            # Loop through all slices
            for slice_idx in range(image.shape[2]):
                image_slice = image[:, :, slice_idx]
                mask_slice = mask[:, :, slice_idx]
                pred_slice = pred[:, :, slice_idx]

                # Save visualization for each slice
                save_path = os.path.join(VISUALS_DIR, f'visualization_{batch_idx // BATCH_SIZE + 1}_image_{i}')
                save_visualization(image_slice, mask_slice, pred_slice, slice_idx, save_path)

        print(f'Saved visualizations for images in batch {batch_idx // BATCH_SIZE + 1}')

print(f"Visualizations saved to: {VISUALS_DIR}")