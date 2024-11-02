import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modules import LesionDetectionModel
from dataset import ISICDataset
import os

# Hyperparameters and configuration
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda'
MODEL_SAVE_PATH = 'model_checkpoints'

# Ensuring the model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Loading Model and Data
model = LesionDetectionModel(model_weights='yolov7.pt', device=DEVICE).model
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Loading training and validation datasets
train_dataset = ISICDataset(
    img_dir='/home/Student/s4760436/recognition/YOLO-47604364/ISIC2018/ISIC2018/ISIC2018_Task1-2_Training_Input_x2',
    annot_dir='/home/Student/s4760436/recognition/YOLO-47604364/ISIC2018/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2',
    mode='train'
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

val_dataset = ISICDataset(

    img_dir='/home/Student/s4760436/recognition/YOLO-47604364/ISIC2018/ISIC2018/ISIC2018_Task1-2_Test_Input',
    mode='test'
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Lists to store loss and accuracy for plotting
train_losses, val_losses = [], []
best_val_loss, patience, trigger_times = float('inf'), 3, 0

def train_one_epoch():
    model.train()
    epoch_loss = 0

    for images, targets in train_loader:
        # Ensure images and targets are moved to the same device
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Forward pass
        outputs = model(images)[0].to(DEVICE)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))
    return train_losses[-1]
        
def validate():
    """
    Validate the model on the validation dataset and return the average loss.
    """
    model.eval()
    val_loss = 0
    num_batches = 0  # Count batches with valid loss

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Check if `data` contains both images and targets
            if isinstance(data, tuple) and len(data) == 2:
                print(f"Batch {batch_idx}: Targets available.")
                images, targets = data
                targets = targets.to(DEVICE)
            else:
                images = data
                targets = None  # No targets in test mode
                print(f"Batch {batch_idx}: No targets.")

            images = images.to(DEVICE)
            outputs = model(images)[0].to(DEVICE)

            # Compute loss only if targets are available
            if targets is not None:
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                num_batches += 1
            else:
                print("Warning: Shape mismatch - skipping batch in validation.")


    avg_val_loss = val_loss / num_batches if num_batches > 0 else None
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss if avg_val_loss is not None else 'N/A'}")
    return avg_val_loss

#Training Loop 
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch()
    val_loss = validate()
    scheduler.step()
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss if val_loss else 'N/A'}")

    # Save model only when validation loss improves
    if val_loss and val_loss < best_val_loss:
        best_val_loss, trigger_times = val_loss, 0
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth"))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

def plot_metrics():
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

plot_metrics()
   