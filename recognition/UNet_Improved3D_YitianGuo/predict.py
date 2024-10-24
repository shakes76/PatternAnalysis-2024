import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MRIDataset, val_transforms  # Ensure these are correctly defined in dataset.py
from modules import UNet3D  # Ensure UNet3D is defined in modules.py
import torch.nn.functional as F
from train import split_data

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
val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model and load trained weights
model = UNet3D().to(args.device)
model.load_state_dict(torch.load(args.model_path))
model.eval()
num_classes = model.out_channels

# Evaluate the model
# Initialize evaluation metrics
dice_scores = {f'Class_{i}': [] for i in range(num_classes)}  # out_chanel is the number of classes

with torch.no_grad():
    for idx, batch in enumerate(val_loader):
        images = batch['image'].to(args.device, dtype=torch.float32)
        labels = batch['label'].to(args.device, dtype=torch.long)

        # Predict
        outputs = model(images)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

        # Calculate Dice coefficient for each class
        for i in range(num_classes):
            pred_i = (preds == i).cpu().numpy()
            label_i = (labels == i).cpu().numpy()
            intersection = np.logical_and(pred_i, label_i).sum()
            union = pred_i.sum() + label_i.sum()
            if union == 0:
                dice = 1.0  # If the denominator is 0, both label and prediction are empty, set Dice to 1
            else:
                dice = 2 * intersection / union
            dice_scores[f'Class_{i}'].append(dice)

# Calculate average Dice coefficient
average_dice = {}
for key, value in dice_scores.items():
    average_dice[key] = np.mean(value)
    print(f"{key}: {average_dice[key]:.4f}")

# Print overall average Dice coefficient
overall_dice = np.mean(list(average_dice.values()))
print(f"Overall Average Dice Score: {overall_dice:.4f}")