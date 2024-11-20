import torch
import numpy as np
import matplotlib.pyplot as plt
from modules import UNet3D  # Import your UNet3D model class
from dataset import *  # Import your dataset handling classes
from torch.cuda.amp import autocast
from matplotlib.colors import ListedColormap

# Set up the device for GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare the model
model = UNet3D(in_channels=1, out_channels=6)

# Load the trained model weights from the specified path
model.load_state_dict(torch.load("/home/Student/s4764092/final.pth"))
model.to(DEVICE)
model.eval()

# Initialize variables for Dice score calculation
NUM_CLASSES = 6
dice_scores = {c: [] for c in range(NUM_CLASSES)}
best_images = {c: (None, -1) for c in range(NUM_CLASSES)}
worst_images = {c: (None, 1) for c in range(NUM_CLASSES)}
total_test_dice = {c: 0 for c in range(NUM_CLASSES)}
global_scores = []
best_case = (None, -1)
worst_case = (None, 1)

# Evaluate the model and compute Dice scores
with torch.no_grad():
    for mri_data, label_data in test_loader:
        mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)
        label_data = label_data.to(DEVICE)

        with autocast():
            outputs = model(mri_data)
            preds = torch.argmax(outputs, dim=1)

            # Compute the Dice score for the per class
            for c in range(NUM_CLASSES):
                pred_c = (preds == c).float()
                label_c = (label_data == c).float()
                intersection = (pred_c * label_c).sum()
                union = pred_c.sum() + label_c.sum()
                dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
                total_test_dice[c] += dice_score

                # Update best and worst images for class 'c' based on the Dice score
                if dice_score > best_images[c][1]:
                    best_images[c] = (mri_data.cpu().numpy(), dice_score.item())
                if dice_score < worst_images[c][1]:
                    worst_images[c] = (mri_data.cpu().numpy(), dice_score.item())

# Calculate average Dice scores for each class and globally
avg_dice = {c: total_test_dice[c] / len(test_loader) for c in total_test_dice}
avg_dice_score = np.mean([avg_dice[c].item() for c in avg_dice])

# Display average Dice scores
for c in range(NUM_CLASSES):
    print(f"Class {c}: Average Dice Score = {avg_dice[c].item():.4f}")
print(f"Overall Average Dice Score: {avg_dice_score:.4f}")

# Set up color map for visualizations
colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple']
cmap = ListedColormap(colors[:NUM_CLASSES])

# Visualization function for original images and predictions
def visualize_image_and_prediction(img, pred, title_img, title_pred, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    img_proj = np.max(img[0, 0, :, :, :], axis=-1)
    pred_proj = np.max(pred[0, :, :, :], axis=-1)

    axs[0].imshow(img_proj, cmap='gray')
    axs[0].set_title(title_img)
    axs[1].imshow(pred_proj, cmap=cmap, interpolation='nearest')
    axs[1].set_title(title_pred)
    plt.savefig(save_path)
    plt.close(fig)

# Visualize the best and worst cases for per class
for c in range(NUM_CLASSES):
    best_img, best_score = best_images[c]
    worst_img, worst_score = worst_images[c]
    if best_img is not None and worst_img is not None:
        best_pred = torch.argmax(model(torch.from_numpy(best_img).to(DEVICE)), dim=1).cpu().numpy()
        worst_pred = torch.argmax(model(torch.from_numpy(worst_img).to(DEVICE)), dim=1).cpu().numpy()
        visualize_image_and_prediction(best_img, best_pred, f'Best Original Image (Class {c}): Dice = {best_score:.4f}', f'Best Prediction (Class {c}): Dice = {best_score:.4f}', f'best_class_{c}.png')
        visualize_image_and_prediction(worst_img, worst_pred, f'Worst Original Image (Class {c}): Dice = {worst_score:.4f}', f'Worst Prediction (Class {c}): Dice = {worst_score:.4f}', f'worst_class_{c}.png')

# Visualize overall best and worst cases
if best_case[0] is not None and worst_case[0] is not None:
    best_pred_global = torch.argmax(model(torch.from_numpy(best_case[0]).to(DEVICE)), dim=1).cpu().numpy()
    worst_pred_global = torch.argmax(model(torch.from_numpy(worst_case[0]).to(DEVICE)), dim=1).cpu().numpy()
    visualize_image_and_prediction(best_case[0], best_pred_global, f'Best Case: Overall Average Dice = {best_case[1]:.4f}', f'Best Prediction: Overall Average Dice = {best_case[1]:.4f}', 'best_global.png')
    visualize_image_and_prediction(worst_case[0], worst_pred_global, f'Worst Case: Overall Average Dice = {worst_case[1]:.4f}', f'Worst Prediction: Overall Average Dice = {worst_case[1]:.4f}', 'worst_global.png')
