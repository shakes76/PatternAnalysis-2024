""" 
Train and test a 2D UNet model.

The performance of the model is evaluated by computing and plotting the
dice scores and similar.

Creates a folder for the model, which contains the saved checkpoint,
the training and testing dice scores, and a log file with periodic
messages from the training and testing.

Modified from Shekhar "Shakes" Chandra:
https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing#scrollTo=w2QhUgaco7Sp

Author:
    Joseph Reid

Functions:
    create_log_file: Create log file to track train/test messages
    write_log_file: Write train/test messages to log file
    train_model: Train UNet on training data
    test_model: Test trained UNet on testing data
    plot_training_metrics: Plot the training loss and dice score
    main: Perform the training and testing and plot the results

Dependencies:
    numpy
    matplotlib
    pytorch
    torchmetrics
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.segmentation import GeneralizedDiceScore

from dataset import ProstateDataset
from modules import UNet

# Prostate dataset classes
CLASSES_DICT = {
    0: "BACKGROUND",
    1: "BODY",
    2: "BONES",
    3: "BLADDER",
    4: "RECTUM",
    5: "PROSTATE",
}

# Hyper-params
NUM_EPOCHS = 24
LEARNING_RATE = 1e-3
BATCH_SIZE = 12

# Model name and directory - Change name for each training
MODEL_NAME = "2D_UNet"
MODEL_DIR = os.path.join(os.getcwd(), "trained_models", MODEL_NAME)

# Paths to model-specific files
MODEL_CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoint.pth")
TRAIN_DICE_SCORE_DIR = os.path.join(MODEL_DIR, "training_dice_score.pth")
TEST_DICE_SCORE_DIR = os.path.join(MODEL_DIR, "testing_dice_score.pth")
LOG_DIR = os.path.join(MODEL_DIR, "log.txt")
# Also print messages when they are written to the log file
VERBOSE = True

# Image directories - Change if data is stored differently
MAIN_IMG_DIR = os.path.join(os.getcwd(), "HipMRI_study_keras_slices_data")
TRAIN_IMG_DIR = os.path.join(MAIN_IMG_DIR, "keras_slices_train")
TRAIN_MASK_DIR = os.path.join(MAIN_IMG_DIR, "keras_slices_seg_train")
TEST_IMG_DIR = os.path.join(MAIN_IMG_DIR, "keras_slices_test")
TEST_MASK_DIR = os.path.join(MAIN_IMG_DIR, "keras_slices_seg_test")


def create_log_file():
    """ Create log.txt in MODEL_DIR to save a log of messages."""
    os.makedirs(MODEL_DIR)
    # Use "x" to verify that MODEL_NAME has been changed
    f = open(LOG_DIR, "x")
    f.close()


def write_log_file(msg: str):
    """ Write msg to new line of log.txt."""
    f = open(LOG_DIR, "a")
    f.write(msg + "\n")
    f.close
    if VERBOSE:
        print(msg)


def train_model(
        model: UNet, 
        loader: DataLoader, 
        criterion: nn.CrossEntropyLoss, 
        optimiser: optim.Optimizer,
        gds: GeneralizedDiceScore,
        device: torch.device
        ) -> tuple[list[float], list[list[float]]]:
    """ 
    Train the model on the training data. 
    
    Parameters:
        model: UNet model to be trained
        loader: Training dataloader
        criterion: Loss function to be optimised
        optimiser: Optimisation algorithm
        gds: Class to calculate dice scores
        device: Device to perform the computations

    Returns:
        list[float]: Cross entropy training losses per epoch
        list[list[float]]: Average dice scores of all classes per epoch
    """
    start_time = time.time()

    # Storage for plotting Cross Entropy Loss and Dice Score per epoch
    train_losses = []
    train_dice_scores = torch.zeros([NUM_EPOCHS, len(CLASSES_DICT)])
    train_dice_scores = train_dice_scores.to(device)

    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_running_loss = 0 # Sum of losses for that epoch
        for i, (images, labels) in enumerate(loader):
            # Note: B=batches, C=classes, H=height, W=width
            images = images.to(device) # Size (B, H, W)
            labels = labels.to(device) # (B, C, H, W)

            images = torch.unsqueeze(images, dim=1) # (B, 1, H, W)
            # Predict and calculate loss
            predicted = model(images) # (B, C, H, W)
            loss = criterion(predicted, labels)
            
            # Optimise model
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_running_loss += loss.item()

            # Undo one hot encoding to get dice score (B, H, W)
            _, predicted = torch.max(predicted, 1)
            _, labels = torch.max(labels, 1)

            # Compute Dice Score per class and add to sum
            dice_score = gds(predicted, labels)
            train_dice_scores[epoch] = torch.add(train_dice_scores[epoch], dice_score)
            
            # Write losses and dice scores to log periodically
            if (i+1) % 100 == 0:
                write_log_file(f"Step: {i+1}/{len(loader)}, Loss: {loss.item()}")
                write_log_file(f"Step: {i+1}/{len(loader)}, Dice score: {(train_dice_scores[epoch] / (i+1))}")

        # Epoch finished. Record average loss for that epoch
        epoch_loss = epoch_running_loss / len(loader)
        write_log_file(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Epoch Loss: {epoch_loss}")
        train_losses.append(epoch_loss)
    
    # Training finished
    write_log_file("Saving the model")
    torch.save(model.state_dict(), MODEL_CHECKPOINT_DIR)
    average_train_dice_scores = train_dice_scores / len(loader)
    torch.save(average_train_dice_scores, TRAIN_DICE_SCORE_DIR)
    average_train_dice_scores = average_train_dice_scores.tolist()
    write_log_file(f"Training Cross Entropy Losses:\n{train_losses}")

    end_time = time.time()
    write_log_file(f"Training took {(end_time - start_time)/60:.1f} minutes")
    return (train_losses, average_train_dice_scores)


def test_model(
        model: UNet, 
        loader: DataLoader, 
        gds: GeneralizedDiceScore,
        device: torch.device
        ) -> list[float]:
    """ 
    Test the model on the testing data. 
    
    Parameters:
        model: UNet model to test with
        loader: Testing dataloader
        gds: Class to calculate dice scores
        device: Device to perform the computations

    Returns:
        list[float]: Final dice scores of all classes
    """   
    # Storage for plotting Cross Entropy Loss and Dice Score
    test_dice_scores = torch.zeros(len(CLASSES_DICT))
    test_dice_scores = test_dice_scores.to(device)
    
    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # Note: B=batches, C=classes, H=height, W=width
            images = images.to(device) # Size (B, H, W)
            labels = labels.to(device) # (B, C, H, W)

            images = torch.unsqueeze(images, dim=1) # (B, 1, H, W)
            predicted = model(images) # (B, C, H, W) 

            # Undo one hot encoding to get dice score (B, H, W)
            _, predicted = torch.max(predicted, 1)
            _, labels = torch.max(labels, 1)

            # Compute Dice Score per class and add to sum
            dice_score = gds(predicted, labels)
            test_dice_scores = torch.add(test_dice_scores, dice_score)

        # Testing finished. Compute and print final average dice scores
        average_test_dice_scores = test_dice_scores / len(loader)
        torch.save(test_dice_scores, TEST_DICE_SCORE_DIR)
        average_test_dice_scores = average_test_dice_scores.tolist()
        for key, label in CLASSES_DICT.items():
            write_log_file(f"Dice score for {label} is {average_test_dice_scores[key]:.4f}")

    return average_test_dice_scores


def plot_training_metrics(loss: list[float], dice_score: list[list[float]]):
    """ 
    Plot the training metrics after completing training.
    
    Parameters:
        loss: List of cross entropy training losses per epoch
        dice_score: Average dice scores of all classes per epoch

    Returns:
        None, plots the cross entropy loss and dice score vs epoch
    """
    # Plot cross entropy losses
    loss_fig = plt.figure()
    plt.plot(loss)
    plt.title('Cross Entropy Training Loss vs Epoch')
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, NUM_EPOCHS), rotation=90)
    # Training prostate dice similarity coefficient
    prostate_dice_scores = []
    for epoch_dice_score in dice_score:
        prostate_dice_scores.append(epoch_dice_score[5])
    dice_fig = plt.figure()
    plt.plot(prostate_dice_scores, label='Training Dice Score')
    plt.axhline(0.75, color='r', label='Target Dice Score')
    plt.title('Prostate Dice Score vs Epoch')
    plt.ylabel('Dice Score')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, NUM_EPOCHS), rotation=90)
    plt.legend()
    # Show and save figures
    plt.show()
    loss_fig.savefig('training_loss_plot')
    dice_fig.savefig('training_dice_score_plot')


def main():
    """ 
    Perform training and testing and plot the results.
    
    Initialises the datasets, dataloaders, loss function and optimiser.
    Also creates the log file and writes to it. Finally, plots the
    training metrics and final dice scores.

    Note that global constants are defined at the top of the script.
    Specifically, directories might need to be changed, and the model
    has to be renamed each time as to not overwrite the previous one.
    """
    # Initialise log file and checks if model has been renamed
    create_log_file()

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        write_log_file("Warning CUDA not Found. Using CPU")

    # Training and testing datasets and dataloaders
    train_dataset = ProstateDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, early_stop=False)
    test_dataset = ProstateDataset(TEST_IMG_DIR, TEST_MASK_DIR, early_stop=False)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    # UNet model
    model = UNet()
    model = model.to(device)
    
    # Weights for loss function from calculate_class_weights in dataset.py
    weights = torch.tensor([0.9679, 0.2466, 1.3312, 8.9319, 44.2907, 39.6628]).to(device)
    # Loss function and optimiser algorithm
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Dice score class to calculate class dice scores
    gds = GeneralizedDiceScore(
        num_classes = len(CLASSES_DICT), 
        per_class = True, 
        weight_type = "linear"
        ).to(device)
    
    # Training, testing, then plotting
    train_losses, train_dice_scores = train_model(model, train_loader, criterion, optimiser, gds, device)
    test_model(model, test_loader, gds, device)
    plot_training_metrics(train_losses, train_dice_scores)


if __name__ == "__main__":
    main()