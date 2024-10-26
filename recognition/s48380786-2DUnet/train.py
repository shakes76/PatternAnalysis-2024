import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import numpy as np

from modules import UNet, dice_coefficient, combined_loss
from dataset import load_and_preprocess_data

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Load data
train_loader, val_loader, test_loader = load_and_preprocess_data()

# Training parameters
num_epochs = 10
train_losses = []
val_losses = []

# Initialize weights for each loss component
weight_ce = 0.5 
weight_dice = 0.5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Remove extra dimension
        labels = torch.squeeze(labels, dim=1)

        # Convert one-hot encoded labels to class indices (Required by Cross Entropy Loss)
        labels = torch.argmax(labels, dim=3)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Debugging: print output and label shapes
        #print(f"Outputs shape: {outputs.shape}")
        #print(f"Labels shape: {labels.shape}")

        #loss = criterion(outputs, labels)
        loss = combined_loss(outputs, labels, weight_ce, weight_dice)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # Calculate average loss for this epoch
    train_losses.append(running_loss / len(train_loader))

    # Validation step (optional but recommended)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            val_labels = torch.squeeze(val_labels, dim=1)  # Remove extra dimension, shape becomes [8, 256, 128, 6]
            val_labels = torch.argmax(val_labels, dim=3)   # Convert one-hot encoding to class indices, shape becomes [8, 256, 128]

            outputs = model(val_images)
            loss = criterion(outputs, val_labels)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")
    # Calculate average validation loss for this epoch
    val_losses.append(val_loss / len(val_loader))


# Path to save the trained model
model_save_path = "/home/Student/s4838078/model_saves"
os.makedirs(model_save_path, exist_ok=True)  # Create directory if it doesn't exist
# Save the model after training
torch.save(model.state_dict(), os.path.join(model_save_path, "unet_model.pth"))
print(f"Model saved to {model_save_path}/unet_model.pth")

# Plot the loss
image_save_path = "/home/Student/s4838078/2DUNet_loss/loss_plot.png"

plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.savefig(image_save_path, dpi=300)

"""
# Evaluate on the test set
model.eval()
dice_scores = []
average_dice_scores = []
num_classes = 6

threshold = 0.5 #0.5
with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        outputs = model(test_images)  # Add batch dimension here
        outputs = (outputs > threshold).float()  # Threshold to binary mask

        # Calculate Dice scores (per-class and average)
        dice_scores_batch, avg_dice_score = dice_coefficient(outputs, test_labels, num_classes)
        dice_scores.extend(dice_scores_batch.tolist())  # Add individual class scores for each sample
        average_dice_scores.append(avg_dice_score)  # Store batch average Dice score

average_dice = np.mean(average_dice_scores)
print(f"Average Dice Coefficient (New): {average_dice}")
"""

# Evaluate on the test set
model.eval()
num_classes = 6
dice_scores_per_class = [0] * num_classes  # Initialize a list to store total Dice scores for each class
class_counts = [0] * num_classes  # To keep track of the number of batches for each class
average_dice_scores = []
dice_scores = []  # Initialize to store individual Dice scores for each batch
threshold = 0.5

with torch.no_grad():
    for test_images, test_labels in test_loader:
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)

        outputs = model(test_images)
        outputs = (outputs > threshold).float()  # Threshold to binary mask

        # Calculate Dice scores (per-class and average)
        dice_scores_batch, avg_dice_score = dice_coefficient(outputs, test_labels, num_classes)
        dice_scores.extend(dice_scores_batch.tolist())
        average_dice_scores.append(avg_dice_score)

        # Accumulate Dice scores for each class
        for i, dice_score in enumerate(dice_scores_batch):
            dice_scores_per_class[i] += dice_score
            class_counts[i] += 1

# Calculate average Dice coefficient across all classes
average_dice = np.mean(average_dice_scores)
print(f"Overall Average Dice Coefficient: {average_dice}")

# Calculate and print the average Dice score per class
average_dice_per_class = [dice_scores_per_class[i] / class_counts[i] for i in range(num_classes)]
for class_idx, avg_dice in enumerate(average_dice_per_class):
    print(f"Average Dice Coefficient for Class {class_idx}: {avg_dice}")
