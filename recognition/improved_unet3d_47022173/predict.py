"""
This file contains code to load a trained model and run predictions on the test set. 

Abdullah Badat (47022173), abdullahbadat27@gmail.com
"""

import torch
from dataset import *
import torch
from torch.utils.data import DataLoader, random_split
import torchio as tio
from utils import *
from modules import *
from monai.losses.dice import DiceLoss
from train import *

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

def predict(
    model_path: str,
    images_path: str, 
    masks_path: str,
    save_path: str
) -> None:
    """
    Evaluate a trained 3D UNet model on the test dataset and save the predictions. It also saves
    predictions as NIfTI files and prints the average Dice score across all classes.

    Parameters:
    - images_path (str): Path to the directory containing the input test images.
    - masks_path (str): Path to the directory containing the ground truth masks for the
    - save_path (str): Path to the directory to save the predictions.
    test images.
    """
    test_transforms = tio.Compose([
        tio.RescaleIntensity((0, 1)),
        tio.Resize((WIDTH,HEIGHT,DEPTH)),
        tio.ZNormalization(),
    ])

    # Load the test dataset and dataloader
    test_dataset = ProstateDataset3D(images_path, masks_path, "test", test_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                 num_workers=NUM_WORKERS)

    # Load the trained model
    model = Modified3DUNet(IN_CHANNELS, N_CLASSES, BASE_N_FILTERS)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    model.eval()

    running_loss = 0.0
    dice_scores = [0] * N_CLASSES
    dice_score_labels = DiceLoss(softmax=True, reduction='none')
    dice_score_mean = DiceLoss(softmax=True)

    with torch.no_grad():
        for batch_idx, (inputs, masks, affines) in enumerate(test_dataloader):  
            inputs, masks = inputs.to(device), masks.to(device)
            
            # One-hot encode the ground truth masks, reshape to [B, C, W, H, D]
            one_hot_masks_3d = F.one_hot(masks, num_classes=N_CLASSES).permute(0, 4, 1, 2, 3)

            # Forward pass
            _, predictions, logits = model(inputs)

            # Save predictions
            save(predictions, affines, batch_idx, save_path)
            
            # Compute the loss
            score = dice_score_labels(logits, one_hot_masks_3d)
            running_loss += dice_score_mean(logits, one_hot_masks_3d)
            for i in range(N_CLASSES):
                dice_scores[i] += 1 - score[0][i].item()

    print("Valid loss: ", float(running_loss) / len(test_dataloader))
    print([float(score / len(test_dataloader)) for score in dice_scores])