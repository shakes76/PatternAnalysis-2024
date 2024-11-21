import argparse

import imageio
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader
from dataset import MRIDataset, val_transforms  # Ensure these are correctly defined in dataset.py
from modules import UNet3D  # Ensure UNet3D is defined in modules.py
import torch.nn.functional as F
from train import split_data
from itertools import cycle

# Parse command-line arguments
parser = argparse.ArgumentParser(description='3D UNet Prediction Script')
parser.add_argument('--model_path', type=str, default='/home/Student/s4706162/best_model.pth')
parser.add_argument('--dataset_root', type=str, default='/home/groups/comp3710/HipMRI_Study_open')
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

# Use the data splitting module
splits = split_data(args.dataset_root, seed=42, train_size=0.6, val_size=0.2, test_size=0.2)
test_image_paths, test_label_paths = splits['test']

# Initialize test dataset
test_dataset = MRIDataset(
    image_paths=test_image_paths,
    label_paths=test_label_paths,
    transform=val_transforms,
    norm_image=True,
    dtype=np.float32
)

# Load data using DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model and load trained weights
model = UNet3D().to(args.device)
model.load_state_dict(torch.load(args.model_path))
model.eval()
num_classes = model.out_channels
print(f"Number of classes: {num_classes}")

# Set up color map for visualization
colors = ['black', 'green', 'orange', 'red', 'blue', 'purple', 'brown', 'cyan', 'magenta', 'yellow']
if num_classes > len(colors):
    # If number of classes exceeds the color list, cycle through colors
    color_cycle = cycle(colors)
    extended_colors = [next(color_cycle) for _ in range(num_classes)]
    cmap = ListedColormap(extended_colors)
else:
    cmap = ListedColormap(colors[:num_classes])

# Initialize best slices for each class
best_slices = {
    f'Class_{i}': {
        'dice': 0.0,
        'image_slice': None,
        'label_slice': None,
        'pred_slice': None
    } for i in range(num_classes)
}

# Initialize dice score storage
dice_scores = {f'Class_{i}': [] for i in range(num_classes)}  # out_channel is the number of classes

# Evaluate the model and track best slices
with torch.no_grad():
    for sample_idx, batch in enumerate(test_loader):
        images = batch['image'].to(args.device, dtype=torch.float32)  # Shape: (B, C, H, W, D)
        labels = batch['label'].to(args.device, dtype=torch.long)     # Shape: (B, C, H, W, D)

        # Perform prediction
        outputs = model(images)  # Shape: (B, num_classes, H, W, D)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # Shape: (B, H, W, D)

        # Move data to CPU and convert to NumPy
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()

        # Transpose to [D, H, W]
        try:
            # Handle labels with different dimensions
            if labels_np.ndim == 5:
                # Shape: [B, C, H, W, D]
                if labels_np.shape[1] == 1:
                    label_volume = np.squeeze(labels_np[0, ...], axis=0)  # Shape: [H, W, D]
                else:
                    # 多通道标签，假设是 one-hot 编码
                    label_volume = np.argmax(labels_np[0, ...], axis=0)    # Shape: [H, W, D]
            elif labels_np.ndim == 4:
                # Shape: [B, H, W, D]
                label_volume = labels_np[0, ...]  # Shape: [H, W, D]
            else:
                raise ValueError(f"Unexpected labels_np dimensions: {labels_np.shape}")

            # Transpose to [D, H, W]
            label_volume = np.transpose(label_volume, (2, 0, 1))  # Shape: (D, H, W)
        except Exception as e:
            print(f"Error transposing label_volume for Sample {sample_idx}: {e}")
            continue

        # Transpose image and prediction volumes
        image_volume = np.transpose(images_np[0, 0, ...], (2, 0, 1))  # Shape: (D, H, W)
        pred_volume = np.transpose(preds_np[0, ...], (2, 0, 1))       # Shape: (D, H, W)

        # Print data shapes and ranges for debugging
        print(f"Sample {sample_idx}: image_volume shape: {image_volume.shape}, label_volume shape: {label_volume.shape}, pred_volume shape: {pred_volume.shape}")
        print(f"Sample {sample_idx}: image_volume min={image_volume.min()}, max={image_volume.max()}")
        print(f"Sample {sample_idx}: label_volume min={label_volume.min()}, max={label_volume.max()}")
        print(f"Sample {sample_idx}: pred_volume min={pred_volume.min()}, max={pred_volume.max()}")

        # Determine number of slices
        num_slices = image_volume.shape[0]

        for slice_idx in range(num_slices):
            image_slice = image_volume[slice_idx, :, :]  # Shape: (H, W)
            label_slice = label_volume[slice_idx, :, :]  # Shape: (H, W)
            pred_slice = pred_volume[slice_idx, :, :]    # Shape: (H, W)

            # Print slice value ranges for the first slice of the first sample
            if sample_idx == 0 and slice_idx == 0:
                print(f"Slice {slice_idx}: image_slice min={image_slice.min()}, max={image_slice.max()}")
                print(f"Slice {slice_idx}: label_slice min={label_slice.min()}, max={label_slice.max()}")
                print(f"Slice {slice_idx}: pred_slice min={pred_slice.min()}, max={pred_slice.max()}")

            for i in range(num_classes):
                pred_i = (pred_slice == i).astype(np.uint8)
                label_i = (label_slice == i).astype(np.uint8)

                intersection = np.logical_and(pred_i, label_i).sum()
                union = pred_i.sum() + label_i.sum()

                if union == 0:
                    dice = 1.0  # If both label and prediction are empty
                else:
                    dice = 2 * intersection / union

                # Store dice score
                dice_scores[f'Class_{i}'].append(dice)

                # Update best slice if necessary
                if (label_i.sum() > 0 or pred_i.sum() > 0) and (dice > best_slices[f'Class_{i}']['dice']):
                    best_slices[f'Class_{i}']['dice'] = dice
                    best_slices[f'Class_{i}']['image_slice'] = image_slice
                    best_slices[f'Class_{i}']['label_slice'] = label_slice
                    best_slices[f'Class_{i}']['pred_slice'] = pred_slice

# Calculate and print average Dice scores
average_dice = {}
for key, value in dice_scores.items():
    average_dice[key] = np.mean(value)
    print(f"{key}: {average_dice[key]:.4f}")

# Print overall average Dice score
overall_dice = np.mean(list(average_dice.values()))
print(f"Overall Average Dice Score: {overall_dice:.4f}")

# Visualize and save the best slices for each class
for class_name, data in best_slices.items():
    if data['image_slice'] is not None:
        image_slice = data['image_slice']
        label_slice = data['label_slice']
        pred_slice = data['pred_slice']
        dice = data['dice']

        # Print slice value ranges for debugging
        print(f"{class_name} - Dice: {dice:.4f}")
        print(f"  Image slice min: {image_slice.min()}, max: {image_slice.max()}")
        print(f"  Label slice min: {label_slice.min()}, max: {label_slice.max()}")
        print(f"  Pred slice min: {pred_slice.min()}, max: {pred_slice.max()}")

        # Normalize image slice to 0-1
        if image_slice.max() > image_slice.min():
            image_slice_normalized = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min())
        else:
            image_slice_normalized = np.zeros_like(image_slice)

        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the original image
        im0 = axes[0].imshow(image_slice_normalized, cmap='gray')
        axes[0].set_title(f'{class_name} - Original Image\nDice: {dice:.4f}')
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Plot the Ground Truth label
        im1 = axes[1].imshow(label_slice, cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[1].set_title(f'{class_name} - Ground Truth')
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Plot the Prediction
        im2 = axes[2].imshow(pred_slice, cmap=cmap, vmin=0, vmax=num_classes-1)
        axes[2].set_title(f'{class_name} - Prediction')
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(f'{class_name}_best_slice_segmentation_result.png')  # Save the visualization
        plt.show()

        # Save the slices as separate PNG files for manual inspection
        image_slice_uint8 = (image_slice_normalized * 255).astype(np.uint8)
        label_slice_uint8 = (label_slice / (num_classes-1) * 255).astype(np.uint8)
        pred_slice_uint8 = (pred_slice / (num_classes-1) * 255).astype(np.uint8)

        imageio.imwrite(f"{class_name}_image_slice.png", image_slice_uint8)
        imageio.imwrite(f"{class_name}_label_slice.png", label_slice_uint8)
        imageio.imwrite(f"{class_name}_pred_slice.png", pred_slice_uint8)

        print(f"Saved {class_name} slices as PNG images.")
    else:
        print(f"{class_name}: No valid slices found.")