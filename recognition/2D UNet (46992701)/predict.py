"""
This file loads the model from the saved checkpoint, runs it on the test set and 
calculates and prints out the DICE score.
"""


from modules import *
from dataset import load_data_2D
import torch
from params import *
import os
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Data Pre-processing

# Normalise images
test_imgs = sorted([os.path.join(TEST_IMG_DIR, img) for img in os.listdir(TEST_IMG_DIR) if img.endswith(('.nii', '.nii.gz'))])
test_imgs = load_data_2D(test_imgs, True)

# One-hot encode the labels
test_labels = sorted([os.path.join(TEST_MASK_DIR, img) for img in os.listdir(TEST_MASK_DIR) if img.endswith(('.nii', '.nii.gz'))])
test_labels = load_data_2D(test_labels, False, True)

testing_set = [ [test_imgs[i], test_labels[i]] for i in range(len(test_imgs))]
test_loader = torch.utils.data.DataLoader(testing_set, BATCH_SIZE, True)

model = UNet().to(device)
checkpoint = torch.load(CHECKPOINT_DIR)
model.load_state_dict(checkpoint['state_dict'])


# Calculate DICE Score on test set
model.eval()
dice_score = dice_score(test_loader, device, model)

print(f'Test Dice Score: {dice_score:.4f}')