#!/usr/bin/env python
""" A collection of methods and tools to train the model.

train.py has methods and tools which are related to the training of model,
some sections are commented out, and are to be enabled to change the training method.
For example, there are a few CRITERIONs, and one of preference is to be enabled to utilise
the loss function of preference.
"""

from dataset import train_dict, train_transforms
from modules import ImprovedUnet
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from time import time
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

__author__ = "Ryuto Hisamoto"

__license__ = "Apache"
__version__ = "1.0.0"
__maintainer__ = "Ryuto Hisamoto"
__email__ = "s4704935@student.uq.edu.au"
__status__ = "Committed"

NUM_EPOCHS = 300
BATCH_SIZE = 2
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
LR_INITIAL = 0.985
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CRITERION = DiceLoss(include_background=False, batch=True).to(DEVICE)
# CRITERION = DiceCELoss(include_background=False, batch = True, lambda_ce = 0.2).to(DEVICE) # Based on Thyroid Tumor Segmentation Report
# CRITERION = DiceFocalLoss(include_background=False, batch = True).to(DEVICE) # Default gamma = 2

def compute_dice_segments(predictions: torch.Tensor, ground_truths: torch.Tensor, device: torch.device | str):
    """The method calculates the dice scores for each segment. The score computed
    inside of the method is independent from the loss used to train the model, hence
    its scores are purely dice coefficients for different segments. Both predictions and 
    ground_truths are required to have the same shape.

    Args:
        predictions (torch.Tensor): Softmax/one-hot encoded predictions from the model.
        ground_truths (torch.Tensor): One-hot encoded labels for an image.
        device (torch.device | str): A device the process is based on, 'cuda' or 'cpu'

    Returns:
        torch.Tensor: _A 0 dimensional tensor in which dice coefficient of a corresponding
        label is stored in its index.
    """

    criterion = DiceLoss(reduction='none', batch=True).to(device)

    num_masks = predictions.size(1)

    segment_coefs = torch.zeros(num_masks).to(device)

    segment_losses = criterion(predictions, ground_truths)

    for i in range(num_masks):
        
        segment_coefs[i] = 1 - segment_losses[i, : , : , : ].item()

    return segment_coefs

def train(model: nn.Module, train_loader: DataLoader, criterion, num_epochs: int, device: torch.device | str):
    """The training method that trains the model with given resources and parameters.

    Args:
        model (nn.Module): A model to train.
        train_loader (DataLoader): DataLoader instance which contains image data and their labels for the model
        to compare its performance against.
        criterion (callable): A loss function the model uses to optimise its performance
        num_epochs (int): The number of epochs to train the model on.
        device (torch.device | str): A device the training is based on. It is highly recommended to use 'cuda' for training.

    Returns:
        tuple: A tuple containing:
            - nn.Module: A trained model
            - np.array: An array of overall dice score for each epoch
            - np.array: An array of segment 0 dice score for each epoch
            - np.array: An array of segment 1 dice score for each epoch
            - np.array: An array of segment 2 dice score for each epoch
            - np.array: An array of segment 3 dice score for each epoch
            - np.array: An array of segment 4 dice score for each epoch
            - np.array: An array of segment 5 dice score for each epoch
    """
    
    # Adam is used as an optimiser
    optimiser = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma = LR_INITIAL)

    model.to(device)
    model.train()

    training_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_0_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_1_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_2_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_3_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_4_dice_coefs = np.zeros(NUM_EPOCHS)
    seg_5_dice_coefs = np.zeros(NUM_EPOCHS)

    scaler = GradScaler(device=DEVICE.type)

    accumuldation_steps = 2

    for epoch in range(num_epochs):
        running_dice = 0.0
        total_segment_coefs = torch.zeros(6, device=device)
        for i, batch_data in enumerate(train_loader):

            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            
            with autocast(device_type=DEVICE.type):
                optimiser.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels) 

            segment_coefs = compute_dice_segments(outputs, labels, device)
            total_segment_coefs += segment_coefs
            scaler.scale(loss).backward()

            if (i + 1) % accumuldation_steps == 0 or i + 1 == len(train_loader): # Gradient Accumulation
                scaler.step(optimiser)
                scaler.update()

            running_dice += 1 - loss.item()

        scheduler.step()

        for i in range(len(total_segment_coefs)):
            print(f"Epoch {epoch + 1} Segment {i} - Training Dice Coefficient: {total_segment_coefs[i] / len(train_loader)}")

        seg_0_dice_coefs[epoch] = (total_segment_coefs[0] / len(train_loader))
        seg_1_dice_coefs[epoch] = (total_segment_coefs[1] / len(train_loader))
        seg_2_dice_coefs[epoch] = (total_segment_coefs[2] / len(train_loader))
        seg_3_dice_coefs[epoch] = (total_segment_coefs[3] / len(train_loader))
        seg_4_dice_coefs[epoch] = (total_segment_coefs[4] / len(train_loader))
        seg_5_dice_coefs[epoch] = (total_segment_coefs[5] / len(train_loader))

        print(f"Epoch {epoch + 1}, Training Overall Dice Coefficient: {running_dice / len(train_loader)}")
        training_dice_coefs[epoch] = (running_dice / len(train_loader))

    return (model, training_dice_coefs, seg_0_dice_coefs, seg_1_dice_coefs,
             seg_2_dice_coefs, seg_3_dice_coefs, seg_4_dice_coefs, seg_5_dice_coefs)

if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# create model. 
model = ImprovedUnet()

print("> Start Training")

start = time()

train_set = Dataset(data=train_dict, transform=train_transforms)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)

# train improved unet
trained_model, training_dice_coefs, seg0, seg1, seg2, seg3, seg4, seg5 = train(model, train_loader, criterion = CRITERION,
                                                            device=DEVICE, num_epochs=NUM_EPOCHS)

end = time()

elapsed_time = end - start
print(f"> Training completed in {elapsed_time:.2f} seconds")

epochs = range(1, NUM_EPOCHS + 1)

# Plots the model's performance during the training.
plt.plot(epochs, training_dice_coefs, label='Training Dice Coefficient')
plt.plot(epochs, seg0, label='Segment 0 Dice Coefficient')
plt.plot(epochs, seg1, label='Segment 1 Dice Coefficient')
plt.plot(epochs, seg2, label='Segment 2 Dice Coefficient')
plt.plot(epochs, seg3, label='Segment 3 Dice Coefficient')
plt.plot(epochs, seg4, label='Segment 4 Dice Coefficient')
plt.plot(epochs, seg5, label='Segment 5 Dice Coefficient')
plt.title(f'Dice Coefficient Over Epochs for {CRITERION}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'unet_dice_coefs_over_epochs_{CRITERION}.png')
plt.close()