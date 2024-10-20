import os
import torch
import matplotlib.pyplot as plt
from dataset import MRIDataset
from modules import UNet3D
from torch.utils.data import DataLoader
from torch.amp import autocast

# Directories and parameters
MODEL_PATH = '/home/Student/s4803414/miniconda3/model/new_model.pth'
IMAGE_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_MRs'
MASK_DIR = '/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only'
# MODEL_PATH = '/Users/shwaa/Desktop/model/new_model.pth'
# IMAGE_DIR = '/Users/shwaa/Downloads/HipMRI_study_complete_release_v1/semantic_MRs_anon'
# MASK_DIR = '/Users/shwaa/Downloads/HipMRI_study_complete_release_v1/semantic_labels_anon'

VISUALS_DIR = '/home/Student/s4803414/miniconda3/visuals'
# VISUALS_DIR = '/Users/shwaa/Desktop/visuals'
BATCH_SIZE = 2
NUM_CLASSES = 6

# Ensure the visuals directory exists
os.makedirs(VISUALS_DIR, exist_ok=True)

# Load dataset (you can use the same MRIDataset class here)
dataset = MRIDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=None, augment=False)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH))
# model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

model.to(device)
model.eval()


def save_visualization(image, mask, pred, save_path):
    import matplotlib.pyplot as plt
    import os

    # Determine the number of slices to visualize
    num_slices = 64  # Change this based on how many slices you want to visualize
    slice_indices = range(min(num_slices, image.shape[2]))  # Ensure we don't go out of bounds

    for slice_idx in slice_indices:
        # Convert tensors to numpy arrays and select the slice
        image_slice = image[:, :, slice_idx].cpu().numpy().squeeze()  # Remove singleton dimensions
        mask_slice = mask[:, :, slice_idx].cpu().numpy().squeeze()    # Remove singleton dimensions
        pred_slice = pred[:, :, slice_idx].cpu().numpy().squeeze()    # Remove singleton dimensions

        # Create the figure for each slice
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image slice
        axes[0].imshow(image_slice, cmap='gray')
        axes[0].set_title(f'Image Slice {slice_idx}')
        axes[0].axis('off')  # Turn off axis for better visualization

        # Plot mask slice
        axes[1].imshow(mask_slice, cmap='viridis')
        axes[1].set_title(f'Mask Slice {slice_idx}')
        axes[1].axis('off')  # Turn off axis for better visualization

        # Plot predicted slice
        axes[2].imshow(pred_slice, cmap='viridis')
        axes[2].set_title(f'Prediction Slice {slice_idx}')
        axes[2].axis('off')  # Turn off axis for better visualization

        # Save the figure
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, f'visualization_slice_{slice_idx}.png'))
        plt.close()



# Prediction loop and saving visualizations
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(data_loader):
        images = images.to(device)
        masks = masks.to(device)

        with autocast(device_type='cuda'):
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # Get the predicted class for each pixel

        # Iterate over the batch and save visualizations
        for i in range(images.size(0)):
            image = images[i]
            mask = masks[i]
            pred = predictions[i]

            save_path = os.path.join(VISUALS_DIR, f'visualization_{batch_idx * BATCH_SIZE + i}.png')
            save_visualization(image, mask, pred, save_path)

        print(f'Saved visualizations for batch {batch_idx + 1}/{len(data_loader)}')

print(f"Visualizations saved to: {VISUALS_DIR}")
