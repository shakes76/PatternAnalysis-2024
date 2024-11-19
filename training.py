import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from modules import LesionDetectionModel
from dataset import ISICDataset
import os

# Hyperparameters and configuration
NUM_EPOCHS = 3  # Number of epochs for testing purposes
BATCH_SIZE = 2  # Smaller batch size for debugging purposes
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = 'model_checkpoints'

# Ensuring the model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Loading Model
print("Loading model...")
model = LesionDetectionModel(model_weights='yolov7.pt', device=DEVICE).model

# Defining Loss Function, Optimizer, and Scheduler
print("Setting up optimizer, loss function, and scheduler...")
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Using ReduceLROnPlateau to dynamically reduce the learning rate based on validation loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Loading complete training dataset
print("Loading dataset...")
complete_dataset = ISICDataset(
    img_dir='/home/Student/s4760436/recognition/YOLO-47604364/ISIC2018/ISIC2018_Task1-2_Training_Input_x2',
    annot_dir='/home/Student/s4760436/recognition/YOLO-47604364/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2',
    mode='train',
    transform=None
)

# Splitting the dataset into training and validation (80/20 split)
train_size = int(0.8 * len(complete_dataset))
val_size = len(complete_dataset) - train_size
train_dataset, val_dataset = random_split(complete_dataset, [train_size, val_size])

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Lists to store loss for plotting
train_losses, val_losses = [], []
best_val_loss, patience, trigger_times = float('inf'), 3, 0

def validate():
    print("Starting validation...")
    model.eval()
    val_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            print(f"Validation Batch {batch_idx}: Processing...")

            images, targets = images.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            outputs = model(images)[0]

            # Debugging output shapes
            print(f"Validation Batch {batch_idx}: Output shape - {outputs.shape}, Target shape - {targets.shape}")

            # Ensure the target shape matches the output shape
            targets = targets.view(targets.size(0), -1, targets.size(-1))
            if targets.size(1) != outputs.size(1):
                print(f"Warning: Validation Batch {batch_idx} - Target size {targets.size()} does not match output size {outputs.size()}. Adjusting target shape...")
                targets = torch.nn.functional.interpolate(targets.permute(0, 2, 1), size=outputs.size(1), mode='nearest').permute(0, 2, 1)

            # Calculate loss
            try:
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Validation Batch {batch_idx}: Error during loss calculation - {e}")

    avg_val_loss = val_loss / num_batches if num_batches > 0 else None
    if avg_val_loss is None:
        print("Validation: No valid batches processed. Check validation data.")
    else:
        print(f"Validation Loss: {avg_val_loss:.4f}")

    val_losses.append(avg_val_loss)
    return avg_val_loss

# Training Loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    model.train()
    epoch_loss = 0

    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Training Batch {batch_idx}: Processing...")
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Forward pass
        outputs = model(images)[0]

        # Debugging output shapes
        print(f"Training Batch {batch_idx}: Output shape - {outputs.shape}, Target shape - {targets.shape}")

        # Apply the loss function
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    val_loss = validate()  # Validate after each epoch

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)

    # Print training and validation loss
    print(f"Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss if val_loss else 'N/A'}")

    # Save model only when validation loss improves
    if val_loss and val_loss < best_val_loss:
        best_val_loss, trigger_times = val_loss, 0
        model_save_path = os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at: {model_save_path}")
    else:
        trigger_times += 1
        print(f"Trigger times: {trigger_times}")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

# Plotting Metrics
def plot_metrics():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    print("Plotting training and validation metrics.")

plot_metrics()
