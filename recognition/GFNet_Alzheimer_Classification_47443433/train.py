from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ADNIDataset
from torch.cuda.amp import GradScaler, autocast
import timm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms

"""
This file contains the training loop for the Alzheimer's Classification model.
"""

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms for the dataset in training and testing
# Train transform
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Convert to 3 channel grayscale
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), # Random crop to 224x224
    transforms.RandomRotation(10), # Rotate image by 10 degrees randomly
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flip image horizontally
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Randomly change brightness and contrast
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # Randomly translate image
    transforms.ToTensor(), # Convert to tensor so that it can be fed to the model
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
])

# Test transform
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.CenterCrop(150), # Center crop to 150x150
    transforms.Resize((224, 224)), # Resize to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Hyperparameters
initial_epochs = 50 # Number of epochs to train for
patience = 10 # Number of epochs to wait before early stopping
learning_rate = 0.0001
accumulation_steps = 2 # Number of steps to accumulate gradients for
best_accuracy = 0.0
no_improvement = 0 # Number of epochs since last improvement

batch_size = 32 

# Datasets and DataLoaders
train_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/train', transform=train_transform)
test_dataset = ADNIDataset(root_dir='/PatternAnalysis-2024/recognition/GFNet_Alzheimer_Classification_47443433/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model, Optimiser, Loss Function, Scheduler and GradScaler
model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=2)
model.head = nn.Linear(model.head.in_features, 2) # Change the head to output 2 classes
model = model.to(device)

# Loss function, optimiser, scheduler and scaler
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = optim.AdamW(model.parameters(), lr=learning_rate)
# Cosine Annealing LR Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=num_epochs) 
scaler = GradScaler()

# Lists to store training and validation losses, validation accuracies and learning rates
training_losses = []
validation_losses = []
validation_accuracies = []
learning_rates = []

# Training Loop
for epoch in range(initial_epochs):
    model.train()
    running_loss = 0.0
    num_batches = 0

    # A single epoch
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        # Zero the parameter gradients
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
        # Backward and optimize
        scaler.scale(loss).backward()
        # Gradient accumulation
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
        
        running_loss += loss.item()
        num_batches += 1
    # Calculate average loss
    avg_loss = running_loss / num_batches
    training_losses.append(avg_loss)

    model.eval()
    val_loss = 0.0
    correct, total = 0, 0
    # Validation loop
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(test_loader)
    validation_losses.append(avg_val_loss)
    accuracy = 100 * correct / total
    validation_accuracies.append(accuracy)
    learning_rates.append(optimiser.param_groups[0]['lr'])

    print(f'Epoch [{epoch+1}/{initial_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
    # Save the model if the validation loss is the best so far
    if avg_val_loss < best_accuracy:
        best_accuracy = avg_val_loss
        torch.save(model.state_dict(), 'best_model.ckpt')
        print(f"New best model saved with val loss: {avg_val_loss:.4f}")
        no_improvement = 0
    else:
        no_improvement += 1
    
    if no_improvement >= patience:
        print("Early stopping triggered.")
        break

    scheduler.step(avg_val_loss)
    model.train()

"""
Plotting
"""
epochs_trained = len(training_losses)

# Training and Validation Loss Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs_trained + 1), training_losses, label="Training Loss")
plt.plot(range(1, epochs_trained + 1), validation_losses, label="Validation Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.savefig('loss_curve.png')

# Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs_trained + 1), validation_accuracies, label="Validation Accuracy", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy over Epochs")
plt.legend()
plt.savefig('validation_accuracy.png')

# Learning Rate Schedule
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs_trained + 1), learning_rates, label="Learning Rate", color="green")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule over Epochs")
plt.legend()
plt.grid()
plt.savefig('learning_rate.png')

# Confusion Matrix
all_labels = []
all_preds = []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['NC', 'AD'])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix on Test Set")
plt.savefig('confusion_matrix.png')