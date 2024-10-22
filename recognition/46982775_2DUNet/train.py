# containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

""" 
Modified from Shekhar "Shakes" Chandra:
https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing#scrollTo=w2QhUgaco7Sp

"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchmetrics.functional import dice
from torchmetrics.segmentation import GeneralizedDiceScore
import numpy as np

from dataset import ProstateDataset
from modules import UNet

# Prostate dataset classes
CLASSES_DICT = {
    0: 'BACKGROUND',
    1: "BODY",
    2: "BONES",
    3: "BLADDER",
    4: "RECTUM",
    5: "PROSTATE",
}

# Hyper-params
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 2

# Model name - Change for each training to save model
MODEL_NAME = "2d_unet_initial"
MODEL_CHECKPOINT1 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', MODEL_NAME, 'checkpoint1.pth')
MODEL_CHECKPOINT2 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', MODEL_NAME, 'checkpoint2.pth')
MODEL_CHECKPOINT3 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', MODEL_NAME, 'checkpoint3.pth')
MODEL_CHECKPOINT4 = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'trained_models', MODEL_NAME, 'checkpoint4.pth')

def train_model(model, loader, criterion, optimiser, device):
    """ Train the model on the train data."""
    start_time = time.time()
    # List to store training loss
    train_losses = []
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_running_loss = 0
        for i, (images, labels) in enumerate(loader):
            images = torch.unsqueeze(images, dim=1)
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f" Step: {i+1}/{len(loader)}, Loss: {loss.item()}")
        epoch_loss = epoch_running_loss / len(loader)
        print(f" Epoch: {epoch}/{NUM_EPOCHS}, Epoch Loss: {epoch_loss}")
        train_losses.append(epoch_loss)

        # Save models
        if epoch + 1 == NUM_EPOCHS / 4:
            print("Saving first checkpoint")
            torch.save(model.state_dict(), MODEL_CHECKPOINT1)
        if epoch + 1 == NUM_EPOCHS / 2:
            print("Saving second checkpoint")
            torch.save(model.state_dict(), MODEL_CHECKPOINT2)
        if epoch + 1 == (3 * NUM_EPOCHS / 4):
            print("Saving third checkpoint")
            torch.save(model.state_dict(), MODEL_CHECKPOINT3)
        if epoch + 1 == NUM_EPOCHS:
            print("Saving fourth checkpoint (final model)")
            torch.save(model.state_dict(), MODEL_CHECKPOINT4)

    end_time = time.time()
    print(f"Training took {(start_time - end_time)/60:.1f} minutes")

def test_model(model, loader, device):
    """ Test the model on the test data."""
    start_time = time.time()
    # Store sum of individual dice scores
    test_dice_score = torch.zeros(len(CLASSES_DICT)).to(device)
    # Class to calculate dice score for each class
    gds = GeneralizedDiceScore(num_classes=len(CLASSES_DICT), per_class=True, 
                                weight_type='linear').to(device)
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # Note: B=batch, C=classes, H=height, W=width
            images = images.to(device) # (B, H, W)
            labels = labels.to(device) # (B, C, H, W) - One hot encoding

            images = torch.unsqueeze(images, dim=1) # (B, 1, H, W)
            predicted = model(images) # (B, C, H, W) 

            # Undo one hot encoding (B, C, H, W) -> (B, H, W)
            _, predicted = torch.max(predicted, 1)
            _, labels = torch.max(labels, 1)

            # Compute Dice Score per class and add to sum
            dice_score = gds(predicted, labels)
            test_dice_score = torch.add(test_dice_score, dice_score)

            if (i+1) % 10 == 0:
                print(f"Step: {i+1}/{len(loader)}, Dice score: {(test_dice_score / (i+1))}")

        # Testing finished. Compute and print final average dice scores
        average_test_dice_score = test_dice_score / len(loader)
        for key, label in CLASSES_DICT.items():
            print(f" Dice score for {label} is {average_test_dice_score[key]:.4f}")

    end_time = time.time()
    print(f"Testing took {(start_time - end_time)/60:.1f} minutes")

# For testing purposes
if __name__ == "__main__":
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    # File paths
    main_dir = os.path.join(os.getcwd(), 'recognition', '46982775_2DUNet', 'HipMRI_study_keras_slices_data')
    train_image_path = os.path.join(main_dir, 'keras_slices_train')
    train_mask_path = os.path.join(main_dir, 'keras_slices_seg_train')
    test_image_path = os.path.join(main_dir, 'keras_slices_test')
    test_mask_path = os.path.join(main_dir, 'keras_slices_seg_test')

    # Datasets
    train_dataset = ProstateDataset(train_image_path, train_mask_path)
    test_dataset = ProstateDataset(test_image_path, test_mask_path)

    # Dataloaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False)

    # Model
    model = UNet(in_channels=1, out_channels=6, n_features=64)
    model = model.to(device)

    # Loss Function and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(model, train_loader, criterion, optimiser, device)
    test_model(model, test_loader, device)