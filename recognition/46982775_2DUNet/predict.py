"""
Load a saved model and test performance on validation data.

The performance of the model is evaluated by the class dice scores
on the predicted segmented data. Also visualises one of the predicted
segments and compares it to the truth.

Author:
    Joseph Reid

Functions:
    main: Compute dice score and plot masks of a random sample

Dependencies:
    matplotlib
    torch
    torchmetrics
"""

import os
import argparse

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchmetrics.segmentation import GeneralizedDiceScore

from dataset import ProstateDataset
from modules import UNet
from train import CLASSES_DICT

# Image directories - Change if data is stored differently
MAIN_IMG_DIR = os.path.join(os.getcwd(), "HipMRI_study_keras_slices_data")
VALIDATION_IMG_DIR = os.path.join(MAIN_IMG_DIR, "keras_slices_validate")
VALIDATION_MASK_DIR = os.path.join(MAIN_IMG_DIR, "keras_slices_seg_validate")


def main(model_name):
    """ 
    Load a saved model and show performance on a random sample.
    
    Loads the model and initialises the validation dataset and laoder.
    Calculates the dice score on a random sample and visualises
    the predicted mask alongside the true mask and the regular image.

    Note that global constants, namely the image directories, are 
    defined at the top of the script and may need to be changed.
    """
    # Device config
    device = torch.device("cpu")

    # Validation dataset and dataloader
    validation_dataset = ProstateDataset(VALIDATION_IMG_DIR, VALIDATION_MASK_DIR, early_stop=False)
    validation_loader = DataLoader(validation_dataset, 1, shuffle=True)

    # Load saved model
    model_dir = os.path.join(os.getcwd(), "trained_models", model_name)
    checkpoint_dir = os.path.join(model_dir, "checkpoint.pth")
    checkpoint = torch.load(checkpoint_dir, weights_only=True)
    model = UNet()
    model.load_state_dict(checkpoint)
    model.eval()

    # Dice score class to calculate class dice scores
    gds = GeneralizedDiceScore(
        num_classes = len(CLASSES_DICT), 
        per_class = True, 
        weight_type = "linear"
        ).to(device)

    with torch.no_grad():
        image, label = next(iter(validation_loader))

        image = torch.unsqueeze(image, dim=1)
        predicted = model(image)

        # Undo one hot encoding to get dice score
        _, predicted = torch.max(predicted, 1)
        _, label = torch.max(label, 1)

        # Compute Dice Score per class
        dice_score = gds(predicted, label)

        # Print dice scores
        dice_score = dice_score.tolist()
        for key, dict_label in CLASSES_DICT.items():
            print(f"Dice score for {dict_label} is {dice_score[key]:.4f}")

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image.squeeze())
        axs[0].set_title('Image')
        axs[1].imshow(label.squeeze())
        axs[1].set_title('True Mask')
        axs[2].imshow(predicted.squeeze())
        axs[2].set_title('Predicted Mask')
        for ax in axs:
            ax.axis("off")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Load given UNet model and visualise predicted masks."
    )
    parser.add_argument('model_name', type=str, help='Model name in folder trained_models')
    args = parser.parse_args()

    main(args.model_name)