# containing the source code for training, validating, testing and saving your model. The model
# should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
# sure to plot the losses and metrics during training

""" 
Modified from Shekhar "Shakes" Chandra:
https://colab.research.google.com/drive/1K2kiAJSCa6IiahKxfAIv4SQ4BFq7YDYO?usp=sharing#scrollTo=w2QhUgaco7Sp

Author:
    Joseph Reid

Functions:
    create_log_file: Create log file to track train/test messages
    write_log_file: Write train/test messages to log file
    train_model: Train UNet on training data
    test_model: Test trained UNet on testing data

Dependencies:
    pytorch
    torchmetrics
"""

import os
import time

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

# Model name - Change for each training to save model and scores
MODEL_NAME = "2d_unet_initial"
MODEL_DIR = os.path.join(os.getcwd(), "recognition", "46982775_2DUNet", "trained_models", MODEL_NAME)
MODEL_CHECKPOINT1 = os.path.join(MODEL_DIR, "checkpoint1.pth")
MODEL_CHECKPOINT2 = os.path.join(MODEL_DIR, "checkpoint2.pth")
MODEL_CHECKPOINT3 = os.path.join(MODEL_DIR, "checkpoint3.pth")
MODEL_CHECKPOINT4 = os.path.join(MODEL_DIR, "checkpoint4.pth")
TRAIN_DICE_SCORE_DIR = os.path.join(MODEL_DIR, "training_dice_score.pth")
TEST_DICE_SCORE_DIR = os.path.join(MODEL_DIR, "testing_dice_score.pth")

# Log file directory to save a log of messages
LOG_DIR = os.path.join(MODEL_DIR, "log.txt")

# Also print messages when they are written to the log file
VERBOSE = True


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
            if (i+1) % 50 == 0:
                write_log_file(f"Step: {i+1}/{len(loader)}, Loss: {loss.item()}")
                write_log_file(f"Step: {i+1}/{len(loader)}, Dice score: {(train_dice_scores / (i+1))}")

        # Epoch finished. Record average loss for that epoch
        epoch_loss = epoch_running_loss / len(loader)
        write_log_file(f"Epoch: {epoch+1}/{NUM_EPOCHS}, Epoch Loss: {epoch_loss}")
        train_losses.append(epoch_loss)

        # Save models
        if epoch + 1 == NUM_EPOCHS / 4:
            write_log_file("Saving first checkpoint")
            torch.save(model.state_dict(), MODEL_CHECKPOINT1)
        if epoch + 1 == NUM_EPOCHS / 2:
            write_log_file("Saving second checkpoint")
            torch.save(model.state_dict(), MODEL_CHECKPOINT2)
        if epoch + 1 == (3 * NUM_EPOCHS / 4):
            write_log_file("Saving third checkpoint")
            torch.save(model.state_dict(), MODEL_CHECKPOINT3)
        if epoch + 1 == NUM_EPOCHS:
            write_log_file("Saving fourth checkpoint (final model)")
            torch.save(model.state_dict(), MODEL_CHECKPOINT4)

    # Training finished. Convert dice scores from tensor to list
    average_train_dice_scores = train_dice_scores / len(loader)
    torch.save(average_train_dice_scores, TRAIN_DICE_SCORE_DIR)
    average_train_dice_scores = average_train_dice_scores.tolist()

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

    start_time = time.time()
    
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

            # Write dice scores to log periodically
            if (i+1) % 50 == 0:
                write_log_file(f"Step: {i+1}/{len(loader)}, Dice score: {(test_dice_scores / (i+1))}")

        # Testing finished. Compute and print final average dice scores
        average_test_dice_scores = test_dice_scores / len(loader)
        torch.save(test_dice_scores, TEST_DICE_SCORE_DIR)
        average_test_dice_scores = average_test_dice_scores.tolist()
        for key, label in CLASSES_DICT.items():
            write_log_file(f"Dice score for {label} is {average_test_dice_scores[key]:.4f}")

    end_time = time.time()
    write_log_file(f"Testing took {(end_time - start_time)/60:.1f} minutes")

    return average_test_dice_scores


# For testing purposes
if __name__ == "__main__":
    # Initialise log file
    create_log_file()

    # Device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        write_log_file("Warning CUDA not Found. Using CPU")

    # Image directories
    main_dir = os.path.join(os.getcwd(), "recognition", "46982775_2DUNet", "HipMRI_study_keras_slices_data")
    train_image_path = os.path.join(main_dir, "keras_slices_train")
    train_mask_path = os.path.join(main_dir, "keras_slices_seg_train")
    test_image_path = os.path.join(main_dir, "keras_slices_test")
    test_mask_path = os.path.join(main_dir, "keras_slices_seg_test")

    # Training and testing datasets
    train_dataset = ProstateDataset(train_image_path, train_mask_path, early_stop=False)
    test_dataset = ProstateDataset(test_image_path, test_mask_path, early_stop=False)

    # Training and testing dataloaders
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

    # UNet model
    model = UNet()
    model = model.to(device)

    # Loss function and optimiser algorithm
    # weights = train_dataset.calculate_class_weights()
    weights = torch.tensor([0.96584141, 0.24800861, 1.30525178, 8.34059394, 47.83551869, 39.09455447])
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Dice score class to calculate class dice scores
    gds = GeneralizedDiceScore(
        num_classes = len(CLASSES_DICT), 
        per_class = True, 
        weight_type = "linear"
        ).to(device)

    # Training
    train_losses, average_train_dice_scores = train_model(model, train_loader, criterion, optimiser, gds, device)
    write_log_file(f"Training Cross Entropy Losses:\n{train_losses}")

    # Testing
    average_test_dice_scores = test_model(model, test_loader, gds, device)
    write_log_file(f"Average Test Dice Scores:\n{average_test_dice_scores}")