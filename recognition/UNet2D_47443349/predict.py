"""
predict.py

Author: Alex Pitman
Student ID: 47443349
COMP3710 - HipMRI UNet2D Segmentation Project
Semester 2, 2024

Contains model testing. Predictions are made on the
test set with dice scores reported. Also visualises
some sample predicted segmentations.
"""

from modules import UNet2D
from dataset import ProstateDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import TEST_IMG_DIR, TEST_MASK_DIR
from utils import dice_score
from utils import N_LABELS, LABEL_NUMBER_TO_NAME
from utils import IMAGES_MEAN, IMAGES_STD
import matplotlib.pyplot as plt


# Samples
N_SAMPLES = 3

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

def run_test(model):
    """
    Predictions are made on the test set with dice scores reported.
    """
    # Data loading
    test_set = ProstateDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transforms=None, early_stop=False, normImage=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Initialisations for Dice score tracking
    total_dice_scores = torch.zeros(N_LABELS).to(DEVICE)
    num_batches = 0

    # Test predictions
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            predictions = model(images)

            # Calculate the Dice score for this batch
            dice_scores = dice_score(predictions, masks)
            total_dice_scores += dice_scores
            num_batches += 1

    # Calculate average Dice score for each class
    average_dice_scores = total_dice_scores / num_batches
    average_dice_scores = average_dice_scores.cpu()

    # Report the average Dice score for each class
    for i, score in enumerate(average_dice_scores):
        print(f"Average Dice Score for {LABEL_NUMBER_TO_NAME[i]}: {score.item():.4f}")

# Plot sample image, predicted mask, and true mask
def plot_sample(image, mask, prediction, idx):
    """
    Plots the input image, true mask, and predicted mask.
    """
    # Get the image, mask, and prediction
    image = image.squeeze().cpu().numpy() # (H, W)
    mask = mask.squeeze().cpu().numpy() # (H, W)
    prediction = prediction.squeeze().cpu().numpy() # (H, W)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(mask, vmin=0, vmax=5)
    axes[1].set_title("True Mask")
    axes[1].axis('off')

    axes[2].imshow(prediction, vmin=0, vmax=5)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    name = "Sample " + str(idx) + ".png"
    plt.savefig(name, bbox_inches='tight', format='png')
    plt.show()

# Run plotting of samples
def run_samples(model):
    """
    Runs visualisation of some sample predicted segmentations.
    """
    # Examples
    examples = ProstateDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transforms=None, early_stop=True, normImage=False)
    examples_loader = DataLoader(examples, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for idx, (image, mask) in enumerate(examples_loader):
            if idx >= N_SAMPLES:
                break
            # Get prediction
            norm_image = (image - IMAGES_MEAN) / IMAGES_STD
            norm_image = norm_image.to(DEVICE)
            prediction = model(norm_image)
            prediction = F.softmax(prediction, dim=1)
            prediction = torch.argmax(prediction, dim=1) # Soft (probability-based) to hard prediction

            # Plot sample
            plot_sample(image, mask, prediction, idx)

def main():
    # Model initialisation and loading
    model = UNet2D(in_channels=1, out_channels=6, initial_features=64, n_layers=4).to(DEVICE)
    model.load_state_dict(torch.load("UNet2D_Model.pth"))

    run_test(model)
    run_samples(model)

if __name__ == "__main__":
    main()