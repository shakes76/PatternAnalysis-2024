import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataset import ISICDataset
from modules import YOLO, YOLO_loss, filter_boxes, single_iou
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 10
learning_rate = 0.001
image_size = 416
batch_size = 10

# Training Data Paths - Adjust directories as needed
mask_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Training_Part1_GroundTruth/'
image_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Training_Data/'
labels = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Training_Part3_GroundTruth.csv'

# Loading Training Dataset and DataLoader
train_dataset = ISICDataset(image_dir, mask_dir, labels, image_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model Initialization
model = YOLO(num_classes=2)
model.to(device)
checkpoint_path = "model.pt"

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = YOLO_loss()

# Learning Rate Scheduler (OneCycleLR) - Adjusts learning rate dynamically
total_step = len(train_dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                steps_per_epoch=total_step, epochs=epochs)

# Training Loop
print("Starting training...")
model.train()
start_time = time.time()

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)  # Move data to device

        # Forward pass
        outputs = model(images)  # Get model predictions
        total_loss = sum(criterion(outputs[a], labels[a]) for a in range(batch_size))  # Calculate batch loss

        # Backward pass and optimization
        optimizer.zero_grad()      # Clear gradients
        total_loss.backward()      # Backpropagation
        optimizer.step()           # Update model parameters
        scheduler.step()           # Adjust learning rate

        # Log training progress every 50 steps
        if (i + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{total_step}], Loss: {total_loss.item():.5f}")
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)

elapsed_time = time.time() - start_time
print(f"Training completed in {elapsed_time:.2f} seconds or {elapsed_time / 60:.2f} minutes.")

# Test Data Paths - Update directories as needed
mask_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Test_v2_Part1_GroundTruth/'
image_dir = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Test_v2_Data/'
labels = '/Users/mariam/Downloads/COMP3710_YOLO/ISIC-2017_Test_v2_Part3_GroundTruth.csv'

# Loading Test Dataset and DataLoader
test_dataset = ISICDataset(image_dir, mask_dir, labels, image_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Testing Loop
print("Starting testing...")
model.eval()
torch.set_grad_enabled(False)  # Disable gradients for testing
start_time = time.time()
total_iou = 0  # Cumulative IoU for averaging
total_step = len(test_dataloader)

for i, (images, labels) in enumerate(test_dataloader):
    images, labels = images.to(device), labels.to(device)  # Move data to device
    outputs = model(images)  # Get model predictions

    # Calculate IoU for each batch
    for a in range(batch_size):
        best_box = filter_boxes(outputs[a])  # Select box with highest confidence
        if best_box is not None:
            best_box = torch.reshape(best_box, (1, 7))  # Reshape to required format
            iou = single_iou(best_box, labels[a, :])  # Calculate IoU between prediction and ground truth
            total_iou += iou[0]  # Accumulate IoU

    # Calculate average IoU for progress monitoring
    average_iou = total_iou / (i + 1)

    # Log testing progress every 50 steps
    if (i + 1) % 50 == 0:
        print(f"Step [{i + 1}/{total_step}], IoU Average: {average_iou:.5f}")

# Calculate total time for testing
elapsed_time = time.time() - start_time
print(f"Testing completed in {elapsed_time:.2f} seconds or {elapsed_time / 60:.2f} minutes.")
