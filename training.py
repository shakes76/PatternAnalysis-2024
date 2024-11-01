import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modules import LesionDetectionModel
from dataset import ISICDataset
import os

os.environ['TORCH_HOME'] = '/home/Student/s4760436/torch_cache'

# Hyperparameters and configuration
NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
DEVICE = 'cpu' if not torch.cuda.is_available() else 'cuda'  # Set to CPU if CUDA is unavailable
MODEL_SAVE_PATH = 'model_checkpoints'

# Ensuring the model save directory exists
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Loading Model and Data
model = LesionDetectionModel(model_weights='yolov7.pt', device=DEVICE).model
criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Loading training and validation datasets
train_dataset = ISICDataset(
    img_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2',
    annot_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2',
    mode='train'
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

val_dataset = ISICDataset(
    img_dir='/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Test_Input',
    mode='test'
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

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
        for data in val_loader:
            images = data[0] if isinstance(data, tuple) else data
            images = images.to(DEVICE)
            outputs = model(images)[0].to(DEVICE)
            val_loss += criterion(outputs, data[1]).item() if isinstance(data, tuple) else 0
            num_batches += 1
    avg_val_loss = val_loss / num_batches if num_batches > 0 else None
    val_losses.append(avg_val_loss)
    return avg_val_loss

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_one_epoch()
    val_loss = validate()
    scheduler.step()
    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss if val_loss else 'N/A'}")

    if val_loss and val_loss < best_val_loss:
        best_val_loss, trigger_times = val_loss, 0
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth"))
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

def plot_metrics():
    """
    Plot training and validation losses over epochs.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch()
        val_loss = validate()

        # Saving the model checkpoint after each epoch
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, f"model_epoch_{epoch+1}.pth"))

    # Plotting the metrics after training
    plot_metrics()

