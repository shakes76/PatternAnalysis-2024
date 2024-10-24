import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from dataset import *
from modules import UNet3D
import numpy as np
from loss import *

# Configure memory management to reduce fragmentation in CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Training hyperparameters
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
ACCUMULATION_STEPS = 4
NUM_CLASSES = 6

# Device setup: use CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")

# Initialize the UNet3D model with 1 input channel and 6 output classes
model = UNet3D(in_channels=1, out_channels=NUM_CLASSES).to(DEVICE)

# Select the Dice loss function for training
criterion = DiceLoss()

# Setup the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()  # For mixed-precision training

# Lists to hold loss and Dice scores for analysis
train_losses = []
val_losses = []
avg_dice_scores = []
class_val_dice_scores = {c: [] for c in range(NUM_CLASSES)}  # Dice scores by class

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()  # Set the model to training mode
    train_loss = 0  # Accumulate losses over the epoch
    optimizer.zero_grad()  # Clear previous gradients

    # Process each batch
    for i, (mri_data, label_data) in enumerate(train_loader):
        mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)  # Adjust data dimensions
        label_data = label_data.to(DEVICE)  # Move labels to the same device

        with autocast():
            outputs = model(mri_data)  # Forward pass
            loss = criterion(outputs, label_data) / ACCUMULATION_STEPS  # Calculate loss

        scaler.scale(loss).backward()  # Backpropagate errors with scaling

        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)  # Update model parameters
            scaler.update()
            optimizer.zero_grad()  # Reset gradients for next accumulation

        train_loss += loss.item() * ACCUMULATION_STEPS  # Update total loss

    avg_train_loss = train_loss / len(train_loader)  # Calculate average loss
    train_losses.append(avg_train_loss)  # Record for later visualization
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Training Loss: {avg_train_loss:.4f}")

    torch.cuda.empty_cache()  # Clear unused memory

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    total_val_dice = {c: 0 for c in range(NUM_CLASSES)}

    with torch.no_grad():  # Disable gradient calculation
        for mri_data, label_data in val_loader:
            mri_data = torch.unsqueeze(mri_data, 1).to(DEVICE)
            label_data = label_data.to(DEVICE)

            with autocast():
                outputs = model(mri_data)
                loss = criterion(outputs, label_data)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)  # Predict classes
                for c in range(NUM_CLASSES):  # Calculate Dice score per class
                    pred_c = (preds == c).float()
                    label_c = (label_data == c).float()
                    intersection = (pred_c * label_c).sum()
                    union = pred_c.sum() + label_c.sum()
                    total_val_dice[c] += (2. * intersection + 1e-6) / (union + 1e-6)

        avg_val_loss = val_loss / len(val_loader)  # Average validation loss
        val_losses.append(avg_val_loss)

        avg_dice = {c: total_val_dice[c] / len(val_loader) for c in total_val_dice}
        avg_dice_score = np.mean([avg_dice[c].item() for c in avg_dice])
        avg_dice_scores.append(avg_dice_score)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Validation Dice Score: {avg_dice_score:.4f}")
        for c in range(NUM_CLASSES):
            class_val_dice_scores[c].append(avg_dice[c].item())
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Class {c} Validation Dice Score: {avg_dice[c].item():.4f}")

    # Visualization of losses and Dice scores at the end of training
    if epoch == NUM_EPOCHS - 1:
        # Visualizing the training and validation losses over each epoch.
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_training_validation_loss.png')
        plt.show()

        # Visualizing the average Dice scores across epochs to monitor performance improvement.
        plt.figure(figsize=(12, 5))
        plt.plot(range(1, len(avg_dice_scores) + 1), avg_dice_scores, label='Average Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Average Dice Score Over Epochs')
        plt.legend()
        plt.savefig(f'epoch_{epoch + 1}_Validation_Dice_Score.png')
        plt.show()

        # Visualizing Dice scores for each class over all epochs to assess class-specific performance.
        plt.figure(figsize=(20, 12))
        for c in range(NUM_CLASSES):
            plt.plot(range(1, len(class_val_dice_scores[c]) + 1), class_val_dice_scores[c],
                     label=f'Class {c} Dice Score')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Class Specific Dice Scores Over Epochs')
        plt.legend(loc='lower right')
        plt.savefig(f'epoch_{epoch + 1}_class_specific_validation_dice_scores.png')
        plt.show()

# Save the final trained model
torch.save(model.state_dict(), 'final.pth')
print("Final model saved.")
