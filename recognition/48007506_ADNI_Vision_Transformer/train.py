"""
train.py

Source code for training, validating, testing and saving the vision transformer.

Author: Chiao-Yu Wang (Student No. 48007506)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from modules import GFNet
from dataset import load_data
from constants import NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, WEIGHT_DECAY, SCHEDULER_FACTOR, SCHEDULER_PATIENCE
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the training and validation losses and accuracies over the epochs.
    """
    # Plot losses for training and validation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies for training and validation
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def train_model():
    """
    Trains the GFNet model using training and validation data, tracks performance metrics (loss and accuracy), 
    and adjusts the learning rate based on validation accuracy. Saves the trained model and plots the 
    training/validation losses and accuracies over the epochs.
    """
    # Set the device (MPS for Apple Silicon, CUDA for GPUs, or CPU fallback)
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")

    # Load the data
    train_loader, val_loader, _ = load_data()

    # Initialise the GFNet model
    model = GFNet(num_classes=2).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler to reduce learning rate if validation accuracy stops improving
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR)

    # Initialize lists to store losses and accuracies
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_train_loss = 0.0
        total_train = 0
        correct_train = 0

        for i, (images, labels) in enumerate(train_loader):
            # Move images and labels to the device
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Backpropagation and optimisation
            optimizer.step()

            # Track training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Print statistics
            running_loss += loss.item()
            epoch_train_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Step [{i + 1}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Store average training loss and accuracy for the current epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation step
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Step the scheduler based on the validation accuracy
        scheduler.step(val_accuracy)

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'Model saved to {MODEL_SAVE_PATH}')

    # Display the losses and accuracies as a plot
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

def validate_model(model, val_loader, criterion):
    """
    Validates the model on the validation set and returns the average validation loss
    and accuracy.
    """
    # Set the device (MPS, CUDA, or CPU)
    device = torch.device("mps" if torch.backends.mps.is_available() 
                          else "cuda" if torch.cuda.is_available() 
                          else "cpu")

    # Set model to evaluation mode
    model.eval()

    correct = 0
    total = 0
    val_loss = 0.0

    # Disable gradient calculation for validation
    with torch.no_grad():
        for images, labels in val_loader:
            # Move images and labels to the selected device
            images, labels = images.to(device).float(), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate validation loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return avg_val_loss, accuracy

if __name__ == '__main__':
    train_model()
