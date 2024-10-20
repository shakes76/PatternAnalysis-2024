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
VISUALS_DIR = '/home/Student/s4803414/miniconda3/visuals'
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
model.to(device)
model.eval()


# Function to overlay predictions on images and save the visualizations
def save_visualization(image, mask, pred, save_path):
    plt.figure(figsize=(12, 4))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(0).cpu().numpy(), cmap='gray')
    plt.title('Original Image')

    # Ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask.squeeze(0).cpu().numpy(), cmap='nipy_spectral')
    plt.title('Ground Truth')

    # Predicted mask
    plt.subplot(1, 3, 3)
    plt.imshow(pred.squeeze(0).cpu().numpy(), cmap='nipy_spectral')
    plt.title('Prediction')

    plt.savefig(save_path)
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
