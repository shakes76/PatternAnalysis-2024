"""
#TODO

@author Damian Bellew
"""

from utils import *
from modules import *
from dataset import *

import torch
import torch.utils
from segmentation_models_pytorch.losses import DiceLoss

def test_model(device, model, test_loader, criterion):
    model.eval()
    dice_score = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Prediction
            outputs = model(images)
            labels = labels.long().view(-1)
            labels = F.one_hot(labels, num_classes=NUM_CLASSES)

            # Compute loss
            loss = criterion(outputs, labels)
            dice_score += loss.item()

    # Compute the average test loss and Dice score
    avg_dice_score = dice_score / len(test_loader)

    print(f'Average Dice Score: {avg_dice_score}')


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU")

# Data loaders
_, test_loader = get_data_loaders()

# Load saved model
model = Unet3D(IN_DIM, NUM_CLASSES, NUM_FILTERS).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

# Test the model
test_model(device, model, test_loader, DiceLoss(mode='multiclass', from_logits=False, smooth=SMOOTH))